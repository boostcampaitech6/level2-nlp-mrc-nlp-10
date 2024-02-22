import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from collections import Counter

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm, trange
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.nn import functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from arguments import DataTrainingArguments, ModelArguments
import torch

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class TFIDFRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, topk_acc_test: bool = False
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)   #질문과 문서의 연관점수와 그 문서의 index반환
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(   #질문과 문서의 연관점수와 그 문서의 index반환
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]) if topk_acc_test == False \
                        else "[TEST_ACC]".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        # print(f'{query_vec.shape=}, {self.p_embedding.T.shape=}')
        # return

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

##########################################################################################################################

class BM25Retrieval:
    def __init__(self, 
                 tokenize_fn, 
                 data_path: Optional[str] = "../data/", 
                 context_path: Optional[str] = "wikipedia_documents.json") -> NoReturn:

        self.data_path = data_path
        self.tokenize_fn = tokenize_fn
        self.tokenized_contexts = None
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        with timer('Tokenizing Context Dataset'):
            self.tokenized_contexts = [self.tokenize_fn(context) for context in self.contexts]  

        with timer("Ready BM25"):
            self.bm25 = BM25Okapi(self.tokenized_contexts)  #위에서 토크나이즈한 문서들을 BM25 점수로 나타내는 부분


    def retrieve(
            self, query_or_dataset: Dataset, topk: Optional[int] = 1, topk_acc_test: bool = False
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert isinstance(query_or_dataset, Dataset), "BM25 Retriever input type is only dataset"

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("Tokenizing Query Dataset"):
            tokenized_query = [self.tokenize_fn(query) for query in query_or_dataset['question']]
            
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
            get_topk = self.bm25.get_top_n(tokenized_query[idx], self.contexts, n=topk) #BM25로 TopK개의 문서를 List로 받는다.
                #["나는 어려서부터 무언가가 . .....", "이순신은 조선의 무신으로.....", .....]   내용물들은 여러 문장들이 있는 문서들이다.
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": " ".join(get_topk) if topk_acc_test == False \
                    else "[TEST_ACC]".join(get_topk),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
        cqas = pd.DataFrame(total)
        return cqas

# class Encoder(AutoModel):
#         def __init__(self, pretrained_model_name, config):
#             super(AutoModel, self).__init__(config)

#             self.model = AutoModel.from_pretrained(pretrained_model_name)#,config)
#             self.init_weights()

#         def forward(self, input_ids,
#                     attention_mask=None, token_type_ids=None):

#             outputs = self.model(input_ids,
#                                 attention_mask=attention_mask,
#                                 token_type_ids=token_type_ids)

#             pooled_output = outputs[1]

#             return pooled_output

class DenseRetrieval:
    def __init__(self,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        pretrained_model_name=None):

        assert pretrained_model_name, "pre_trainedmodel을 입력하세요, --model  "

        # self.p_encoder=None
        # self.q_encoder=None
        self.data_path = data_path
        self.encoder_trained=False
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.del_model_name=pretrained_model_name[pretrained_model_name.index('/')+1:]

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )
            
    def create_encoder(self):
        pretrained_model_name=self.pretrained_model_name
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.p_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.q_encoder = AutoModel.from_pretrained(pretrained_model_name)

        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()


    def encoders_train(self, args, dataset, overflow_mapping):
        assert self.p_encoder and self.q_encoder, "인코더를 생성하세요. \"create_encoder(pretrained_model_name)\""

        self.q_model_dir = f"{self.pretrained_model_name}_q_model.pt"
        self.p_model_dir = f"{self.pretrained_model_name}_p_model.pt"
        q_model_dir=self.q_model_dir
        p_model = self.p_encoder
        q_model = self.q_encoder
        data_args=DataTrainingArguments
        
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # self.context_ids = self.tokenizer(self.contexts, max_length = data_args.max_seq_length, padding="max_length", return_tensors='pt',
        #                 truncation=True, stride = data_args.doc_stride, return_overflowing_tokens=True, 
        #                 return_token_type_ids=False)['overflow_to_sample_mapping'].to("cuda")

        # Start training!
        print(f'\n!!!!!!!!!!!!!Train Start!!!!!!!!!!!!!!!!\n')
        global_step = 0

        p_model.zero_grad()
        q_model.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:    #epoch
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):   #batch가 train loader에서 batch_size만큼씩 받은 데이터
                q_model.train()
                p_model.train()
                len_batch = args.per_device_train_batch_size

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                curr_contexts_ids = batch[6]

                self.p_inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                            }

                q_inputs = {
                            'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]
                            }

                p_inputs = self.p_inputs
                
                p_outputs = p_model(**p_inputs)['pooler_output']  # (batch_size, emb_dim)
                q_outputs = q_model(**q_inputs)['pooler_output']  # (batch_size, emb_dim)

                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                #targets = torch.arange(0, args.per_device_train_batch_size).long()
                #[0,1,2,3]  [4,5,5,6]
                count = Counter([id.item() for id in curr_contexts_ids])   #ex) [2,1,1,2] => {2:2, 1:2}
                targets = []
                # for idx, n in count.items():
                #     start_idx = len(targets)
                #     val=1/n
                #     target = [0]*args.per_device_train_batch_size
                #     target[start_idx:start_idx+n]=[val]*n
                #     for _ in range(n):
                #         targets.append(target)
                # print(count)
                for i in range(args.per_device_train_batch_size):
                    target = [0]*args.per_device_train_batch_size
                    curr_contexts_id = curr_contexts_ids[i]
                    # print(curr_contexts_id)
                    # print(count[curr_contexts_id.item()])
                    # print()
                    val = 1/count[curr_contexts_id.item()]
                    for j in range(args.per_device_train_batch_size):
                        if curr_contexts_id == curr_contexts_ids[j]:
                            target[j]=val
                    
                    targets.append(target)

                targets = torch.tensor(targets).to("cuda")  #batch X batch
                # print(curr_contexts_ids)
                # for i in range(len(targets)):
                #     print(targets[i])

                # print("\n\n\n")
                
                sim_scores = -F.log_softmax(sim_scores, dim=1)
                loss = torch.sum(targets*sim_scores)/args.per_device_train_batch_size
                # loss = F.nll_loss(sim_scores, targets.to('cuda'))

                if step%50==0:
                    print(f' Loss : {loss}')

                loss.backward()
                optimizer.step()
                scheduler.step()
                q_model.zero_grad()
                p_model.zero_grad()
                global_step += 1

        print(f'\n!!!!!!!!!!!!!Train Finish!!!!!!!!!!!!!!!!\n')
        self.encoder_trained=True
        self.p_encoder, self.q_encoder = p_model, q_model

        del_model_name=self.q_model_dir[self.q_model_dir.index('/')+1:]
        torch.save(self.q_encoder, f'./models/Dense/{del_model_name}')
        torch.save(self.p_encoder, f'./models/Dense/{del_model_name}')
        print(f'\n!!!!!!!!!!!!!Model Save!!!!!!!!!!!!!!!!\n')

    def get_dense_embeddings(self):
        file_name = 'Dense/p_dense_embedding.bin'
        file_path = os.path.join(self.data_path, file_name)
        del_model_name=self.pretrained_model_name[self.pretrained_model_name.index('/')+1:]

        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                self.p_embedding = pickle.load(f)
            with open(self.data_path+"/Dense/context_ids.bin", "rb") as f:            
                self.context_ids = pickle.load(f)

            print("\nEmbedding load!\n")
            self.q_encoder = torch.load(f'./models/Dense/{self.del_model_name}_q_model.pt').to("cuda")

        else:
            if os.path.isfile(f'./models/Dense/{self.pretrained_model_name}_q_model.pt') and os.path.isfile(f'./models/Dense/{self.pretrained_model_name}_p_model.pt'):
                self.p_encoder = torch.load(f'./models/Dense/{del_model_name}_p_model.pt').to("cuda")
                self.q_encoder = torch.load(f'./models/Dense/{del_model_name}_q_model.pt').to("cuda")
                print("\nEncoder load!\n")

            else:
                data_args=DataTrainingArguments
                datasets = load_from_disk("../data/train_dataset")
                training_dataset = datasets['train']

                p_seqs = self.tokenizer(
                    training_dataset['context'], 
                    max_length = data_args.max_seq_length,
                    padding="max_length", 
                    return_tensors='pt',
                    truncation=True,
                    stride = data_args.doc_stride,
                    return_overflowing_tokens=True,
                    #return_token_type_ids=False,
                    )

                #print(p_seqs["overflow_to_sample_mapping"][29:36])
                self.overflow_mapping = p_seqs["overflow_to_sample_mapping"]
                new_q = [training_dataset['question'][i] for i in tqdm(p_seqs["overflow_to_sample_mapping"], desc="matching Q&P")]  #query들을 문서에 맞게 추가해줌
                #print(new_q[29:36])

                q_seqs = self.tokenizer(
                    new_q,
                    max_length = data_args.max_seq_length,
                    padding="max_length",
                    return_tensors='pt', 
                    truncation=True, 
                    stride = data_args.doc_stride,
                    #return_overflowing_tokens=True,
                    #return_token_type_ids=False,
                    )

                self.train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                    q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'], self.overflow_mapping)
                
                self.create_encoder()
                args = TrainingArguments(
                output_dir="dense_retireval",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=6,
                per_device_eval_batch_size=6,
                num_train_epochs=3,
                weight_decay=0.01
                )
                print("\nencoders_training!\n")
                self.encoders_train(args, self.train_dataset, self.overflow_mapping)
                
            # p_tokens = self.tokenizer(
            #     self.contexts, 
            #     max_length = data_args.max_seq_length,
            #     padding="max_length", 
            #     return_tensors='pt',
            #     truncation=True,
            #     stride = data_args.doc_stride,
            #     return_overflowing_tokens=True,
            #     #return_token_type_ids=False,
            #     )

            # self.context_ids = p_tokens.pop('overflow_to_sample_mapping')
            # p_tokens.to("cuda")###################요기가 아무래도 문제인듯?

            # self.p_embedding = self.p_encoder(**p_tokens)['pooler_output'].to("cpu")
            with torch.no_grad():
                self.p_encoder.eval()
                self.q_encoder.eval()

                self.p_embedding = []
                self.context_ids = []
                for i in trange(len(self.contexts), desc="making p_embedding"):
                    p = self.tokenizer(self.contexts[i], max_length = data_args.max_seq_length, padding="max_length", return_tensors='pt',
                        truncation=True, stride = data_args.doc_stride, return_overflowing_tokens=True,
                        return_token_type_ids=False,
                        ).to("cuda")
                    
                    context_ids = p.pop('overflow_to_sample_mapping').to("cpu")
                    p_emb=self.p_encoder(**p)['pooler_output'].to("cpu").numpy()

                    # print(len(self.tokenizer.tokenize(self.contexts[i])),"!!!")
                    # print(f'{context_ids}')
                    
                    self.p_embedding.extend(p_emb)
                    self.context_ids.extend([i for _ in range(len(context_ids))])

            self.p_embedding = torch.tensor(self.p_embedding).squeeze()
            self.context_ids = torch.tensor(self.context_ids).squeeze()
            print(f'\n{self.p_embedding.shape=}')
            print(f'\n{self.context_ids.shape=}')

            print("\n!!!!!!!!!!p_embed!!!!!!!!!!!\n")
            #print(p_tokens.keys())
            self.p_encoder.to("cpu")
            self.q_encoder.to("cuda")
            
            with open(file_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            
            with open(self.data_path+"/Dense/context_ids.bin", "wb") as f:
                pickle.dump(self.context_ids, f)
            
            print("\nEmbedding Save!\n")
        
        self.p_embedding
        self.context_ids.to("cuda")

        #테스트!!!!!!!!!!!!!!!!!!
        
        print(f'\nTest Start!!!!!!!!!!\n')
        datasets = load_from_disk("../data/train_dataset")
        training_dataset = datasets['train']
        for i in random.sample(range(3000), 10):
            q = self.tokenizer(training_dataset[i]['question'], max_length = 512, padding="max_length", return_tensors='pt',
                truncation=True, stride = 128, return_overflowing_tokens=False,
                return_token_type_ids=False,
                ).to("cuda")
            
            q_embedding = self.q_encoder(**q)['pooler_output'].to("cuda")
            score = torch.matmul(q_embedding, self.p_embedding.T.to("cuda")).squeeze()

            indice = torch.argsort(score, dim=-1).to("cpu")
            print(f"{indice.shape=}")
            indice = indice.tolist()

            print(f"\n question : {training_dataset[i]['question']}")
            print(f" ground_truth : {training_dataset[i]['context']}")
            print(f"\n TopK : ")
            for i in range(10):
                print(" ", self.contexts[self.context_ids[indice[i]]][:50])
            print()
        
        #테스트!!!!!!!!!!!!!!!!!!
    
    def retrieve(
        self, query_or_dataset, topk: Optional[int] = 1, topk_acc_test: bool = False):
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        self.get_dense_embeddings()
        data_args=DataTrainingArguments
        
        # with open(self.data_path+"/Dense/context_ids.bin", "rb") as f:
        #     self.context_ids = pickle.load(f)

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        
        if isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                # doc_scores, doc_indices = self.get_relevant_doc_bulk(   #질문과 문서의 연관점수와 그 문서의 index반환
                #     query_or_dataset["question"], k=topk)

                with torch.no_grad():
                    self.q_encoder.eval()

                    query_embed=[]
                    for i in trange(len(query_or_dataset["question"]), desc = "making query_embedding"):
                        q = self.tokenizer(query_or_dataset["question"][i], max_length = data_args.max_seq_length, padding="max_length", return_tensors='pt',
                            truncation=True, stride = data_args.doc_stride, return_overflowing_tokens=True,
                            return_token_type_ids=False, 
                            ).to("cuda")
                        
                        q_ids = q.pop('overflow_to_sample_mapping').to("cpu")
                        q_emb = self.q_encoder(**q)['pooler_output'].to("cpu").numpy()
                        query_embed.extend(q_emb)

                query_embed = torch.tensor(query_embed).squeeze()
                doc_scores = torch.matmul(query_embed, self.p_embedding.T.to("cpu"))
                print(f'\n!!!!score_shape : {doc_scores.shape}!!!!!\n')
                indices = torch.argsort(doc_scores, dim=-1, descending=True).squeeze().tolist()
                doc_indices = [list(set([self.context_ids[i] for i in row_indices[:topk]])) for row_indices in indices]    #context의 index로 바꾸기   
                print(f'\n{topk=}doc_indices shape : [{len(doc_indices)}, {len(doc_indices[0])}]!!!!!\n')

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]) if topk_acc_test == False \
                        else "[TEST_ACC]".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    

##########################################################################################################################

if __name__ == "__main__":

    import argparse
    print("\n", "!!!!!retrieval start!!!!!\n")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", metavar="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(    ############토크나이저 바꾸는곳
        "--model_name_or_path",
        default= ModelArguments.retrieval_tokenizer_name, #tokenizer name
        metavar="pretrained Model",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", metavar="../data", type=str, help="")
    parser.add_argument(
        "--context_path",  default="wikipedia_documents.json", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, metavar=False, type=bool, help="")
    
    # Add argparser for Top-k Test
    
    ######################################################
    #parser.add_argument("--model", default='uomnf97/klue-roberta-finetuned-korquad-v2', metavar="pretrained_model_name", type=str, help='dense의 tokenizer')
    parser.add_argument("--topk", default=20, metavar=10, type=int, help='topk')
    parser.add_argument("--method", default="BM25", metavar="BM25", type=str, help='Retrieval Method, BM25, TF-IDF, Dense')
    parser.add_argument("--test", default=True, metavar=False, type=bool, help='topk acc test')
    ######################################################
    
    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,) #???????????????????????????????????????
    assert args.method in ['BM25', 'TF-IDF', 'Dense'], "Check retrieval method in ['BM25', 'TF-IDF', 'Dense']"

    if args.method == 'TF-IDF':
        retriever = TFIDFRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.get_sparse_embedding()

    elif args.method == 'BM25':
        retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path
        )
    
    elif args.method == 'Dense':
        retriever = DenseRetrieval(
            data_path=args.data_path,
            context_path=args.context_path,
            pretrained_model_name=args.model_name_or_path
        )
    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        if args.test:
            with timer(f"Testing {args.method} Top-{args.topk} acc"):
                if args.method == 'TF-IDF':
                    df = retriever.retrieve(query_or_dataset=full_ds,
                                            topk = args.topk,
                                            topk_acc_test=args.test)
                    
                    correct_cnt = 0
                    pbar = tqdm(range(len(df)))
                    
                    for i in pbar:
                        original_context = df.iloc[i, 3]
                        for retrieval_context in df.iloc[i, 2].split('[TEST_ACC]'):
                            if original_context == retrieval_context:
                                correct_cnt += 1
                                break
                        pbar.set_description(f'Method:{args.method}, Top-{args.topk} ACC: {correct_cnt/(i+1)}, correct: {correct_cnt}, check: {i+1}')


                elif args.method == 'BM25':
                        """BM25Retriever는 시간 효율상 아래와 같은 방법으로 해야 개수에 따른 acc 변화과정을 볼 수 있음."""
                        tokenized_query = [retriever.tokenize_fn(query) for query in full_ds['question']]
                        
                        correct_cnt = 0
                        pbar = tqdm(zip(tokenized_query, full_ds['context']))
                        
                        for i, (query, original_context) in enumerate(pbar):
                            topn_docs = retriever.bm25.get_top_n(query, retriever.contexts, n=args.topk)
                            for retrieval_context in topn_docs:
                                if original_context == retrieval_context:
                                    correct_cnt += 1
                                    break
                            pbar.set_description(f'Top-{args.topk} ACC: {correct_cnt/(i+1)}, correct: {correct_cnt}, check: {i+1}')

                elif args.method == 'Dense':
                    df = retriever.retrieve(query_or_dataset=full_ds, topk = args.topk, topk_acc_test=args.test)
                    
                    correct_cnt = 0
                    pbar = tqdm(range(len(df)))
                    
                    for i in pbar:
                        original_context = df.iloc[i, 3]
                        for retrieval_context in df.iloc[i, 2].split('[TEST_ACC]'):#########
                            if original_context == retrieval_context:
                                correct_cnt += 1
                                break
                        pbar.set_description(f'Method:{args.method}, Top-{args.topk} ACC: {correct_cnt/(i+1)}, correct: {correct_cnt}, check: {i+1}')
