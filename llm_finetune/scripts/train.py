# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
from typing import Optional
import os
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.modeling_utils import unwrap_model
from mlperf_logging_utils import MLPerfCallback,LoraLogger,submission_info,general_info,optimization_info
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from datasets import load_dataset
import numpy as np
import functools
from utils import (
    create_and_prepare_model,
    world_size_from_yaml,
    training_step,
    SaveDeepSpeedPeftModelCallback,
    peft_module_casting_to_bf16,
)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """


    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )

    dataset_path: Optional[str] = field(
        default='./dataset.npy',
        metadata={"help": "The path to the downloaded dataset."},
    )
    config_path: Optional[str] = field(
        default="./configs/default_config.yaml",
        metadata={"help": "path to model config"},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    save: Optional[bool] = field(
        default=False,
        metadata={"help": "Save model after training"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store downloaded dataset from huggingface.co"},
    )
    target_eval_loss: float = field(
        default=0.92, metadata={"help": "target eval loss - NOT FINAL."}
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    num_workers: int = field(
        default=4, metadata={"help": "Number of dataset workers to use."}
    )

    dataset_config_name: Optional[str] = field(default="gov_report")


def main(script_args, training_args):
    loralogger=LoraLogger(target_eval_loss=script_args.target_eval_loss)
    submission_info(loralogger,
                    submission_benchmark="llm-finetuning",
                    submission_division="Closed",
                    submission_org="referece",
                    submission_platform="referece",
                    submission_poc_name="referece",
                    submission_poc_email="referece",
                    submission_status="referece")
                    submission_status="referece")

    # training arguments
    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"
        and script_args.use_peft_lora
    )

    if training_args.local_rank == 0:
        print(f"Script args: {script_args}")
        print(f"Training args: {training_args}")
    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True
    save_strategy = "steps"


    # model
    model, peft_config, tokenizer = create_and_prepare_model(
        script_args, training_args)
    model.config.use_cache = False

    # datasets
    #dataset = load_dataset("regisss/scrolls_gov_report_preprocessed_mlperf_2")
    dataset = np.load(script_args.dataset_path,allow_pickle=True).tolist()
    train_dataset, eval_dataset = dataset["train"], dataset["validation"]
    #train_dataset, eval_dataset = create_datasets(tokenizer, args)

    world_size = world_size_from_yaml(script_args.config_path)
    general_info(loralogger,training_args,world_size=world_size,eval_samples=len(eval_dataset),train_samples=len(train_dataset))
    optimization_info(loralogger,training_args)


    # trainer
    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MLPerfCallback(loralogger)],
    )
    trainer.training_step = functools.partial(training_step, trainer)
    trainer.accelerator.print(f"{trainer.model}")
    if script_args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if is_deepspeed_peft_enabled:
        trainer.add_callback(
            SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps)
        )

    if script_args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, training_args)

    # train
    trainer.train()

    # Save the PEFT adapter on main process
    if trainer.args.process_index == 0:
        if training_args.push_to_hub:
            print("Push to hub...")
            trainer.push_to_hub()
            if training_args.use_peft_lora:
                trainer.model.push_to_hub(training_args.output_dir)
        elif script_args.save:
            print("Save model...")
            unwrap_model(trainer.model).save_pretrained(
                training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GaudiTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args)
