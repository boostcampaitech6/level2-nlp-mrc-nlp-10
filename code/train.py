
import logging
import os
import sys
import random
import numpy as np
import torch
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions


seed = 2024
deterministic = False

random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    
    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    ####################################################
    training_args.per_device_train_batch_size=15
    training_args.per_device_eval_batch_size=15
    print("batch_size : ", training_args.per_device_train_batch_size)
    print("Epoch : ", training_args.num_train_epochs)
    # training_args.fp16 = True
    # print(training_args.per_device_train_batch_size)
    ####################################################
    

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
    
    datasets = load_from_disk(data_args.dataset_name)
    print("\ndatasets : ")
    print(datasets)

    if model_args.use_extraction:
        # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
        # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name is not None
            else model_args.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
            # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
            # rust version이 비교적 속도가 빠릅니다.
            use_fast=True,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        output_dir=training_args.output_dir
        num_train_epochs=training_args.num_train_epochs
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            predict_with_generate=True,
            num_train_epochs=num_train_epochs,
            save_strategy='epoch',
            save_total_limit=2
        )
        training_args.predict_with_generate = True
        training_args.num_train_epochs = 5
        training_args.per_device_train_batch_size = 5
        training_args.per_device_eval_batch_size = 5
        
        # generation model을 불러옵니다.
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path
            if "train_dataset" in model_args.model_name_or_path
            else model_args.model_for_generation,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path
            if "train_dataset" in model_args.model_name_or_path
            else model_args.model_for_generation,
            use_fast=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path
            if "train_dataset" in model_args.model_name_or_path
            else model_args.model_for_generation,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        if model_args.use_extraction:
            print('...run extraction')
            run_mrc_extract(data_args, training_args, model_args, datasets, tokenizer, model)
        else:
            print('...run generation')
            run_mrc_generate(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc_extract(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]    #질문
    context_column_name = "context" if "context" in column_names else column_names[1]   #본문
    answer_column_name = "answers" if "answers" in column_names else column_names[2]    #{answer_start : , text : } 형태

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"    #tokenizer.padding_side가 "right"라면 pad_on_right가 True값이 되는 구조

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first", #질문과 문서 중 문서에서 짜르기
            max_length=max_seq_length,
            stride=data_args.doc_stride,    #짜른 문서에서 겹치는 부분 수
            return_overflowing_tokens=True, #문서가 짤리면 같은 문서끼리는 같은 숫자 반환 ex)[0,0,1,2,3,3]  0번문ㄴ서 3번문서가 길어서 짤린것
            return_offsets_mapping=True,    #토큰화 된 입력값의 문자단위의 위치끼리 묶어줌/ CLS토큰은 (0,0), 그다음 토큰이 3글자토큰이면 (0,3)
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        #{"input_dis" : [....], "Attention_Mask" : [.....], "offset_mapping":[(0,0), (1,4),.....], "overflow_to_sample_mapping" : [0,0,1,2,2,3,...] }

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):    #offsets는 위의 토큰화한 문서1개당 1개의 list로 반환되므로 i는 몇번째 문서인지, offsets는 그 문서에서 token의 글자수에 따른 위치
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index, 제일 앞에있어서 0

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)   #token_type_ids랑 똑같음. i번쨰 문서에서 질문과 문ㄴ서 구문

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]    #몇번째 문서인지(짤렸을때 짤린문서라는거 표시) [0,0,1,2,2,3] =>0번ㄴ과 2번 문서 짤린문서로 저장했단 뜻
            answers = examples[answer_column_name][sample_index]    #examples는 질문, 문서set / 이 데이터셋에서 answer_column_name은 "answer"
            #위의[0,0,1,2,2,3]의 예에서 첫번쨰 두번쨰 문서는 0번 질문 문서 세트인데 이 세트에서의 answer을 받기 위한것

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):    #sequence_ids(=token_type_ids)에서 1은 [SEP]뒤 즉, 문서를 뜻함
                    token_start_index += 1  
                    #질문 문서 인풋에서 문서의 시작점
                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                     #질문 문서 인풋에서 문서의 끝점

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):  #정답의 시작위치(문자1개씩 기준)와 끝위치가 문서의 시작과 끝 사이에 있지 않다면
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:   #문서 안에 있으면
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def run_mrc_generate(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    # pad_on_right = tokenizer.padding_side == "right"

    max_target_length = 128
    num_beams = 2
    
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train, eval preprocessing / 전처리를 진행합니다.
    def preprocess_function(examples):
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )

        # targets(label)을 위해 tokenizer 설정
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    import nltk
    nltk.download('punkt')

    # Post-processing:
    def postprocess_text(preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = load_metric("squad")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

    # Trainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_target_length,
            num_beams=num_beams,
            metric_key_prefix="eval",
        )

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
