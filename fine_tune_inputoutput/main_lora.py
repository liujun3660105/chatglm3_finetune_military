import logging
import sys
import os
import json
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trainer import LoRATrainer
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from data_preprocess import sanity_check, InputOutputDataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# 初始化日志记录
logger = logging.getLogger(__name__)
# 禁用wangdb
os.environ["WANDB_DISABLED"] = "true"

def setup_logger(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 配置huggingface的日志记录
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def load_lora_model(model_args, peft_args):
    # 加载预训练的chatglm3-6b的tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = model.half()
    # 使用peft库配置lora参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, peft_config).to(model_args.device)
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    return tokenizer, model

def main():
    # 解析传入的命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    # 初始化工作
    setup_logger(training_args)
    set_seed(training_args.seed)
    tokenizer, model = load_lora_model(model_args, peft_args)
    model.print_trainable_parameters()
    # 准备训练数据集并处理成所需格式
    if training_args.do_train:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            if data_args.train_file.endswith(".json"):
                train_data = json.load(f)
            elif data_args.train_file.endswith(".jsonl"):
                train_data = [json.loads(line) for line in f]

        train_dataset = InputOutputDataset(
            train_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )

        #if training_args.local_rank < 1:
        #    sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)
    if training_args.do_eval:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            if data_args.train_file.endswith(".json"):
                eval_data = json.load(f)
            elif data_args.train_file.endswith(".jsonl"):
                eval_data = [json.loads(line) for line in f]

        eval_dataset = InputOutputDataset(
            eval_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
        # 将数据集中样本批处理成张量
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )
    # 配置trainer，LoRATrainer中实现仅保存LoRA参数的功能
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # 开始训练
    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()
