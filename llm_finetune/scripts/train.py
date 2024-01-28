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
from typing import Optional

from datasets import load_dataset
from mlperf_logging_utils import LoraLogger, MLPerfCallback
from transformers import HfArgumentParser, Trainer, TrainingArguments
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from utils import create_and_prepare_model, peft_module_casting_to_bf16


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """


    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "lora dropout is a fixed to 0.1 in closed submission"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "lora rank is a fixed to 16 in closed submission"})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    max_seq_length: Optional[int] = field(default=8192)
    model_path: Optional[str] = field(
        default="./llama-v2-fused-qkv",
        metadata={"help": "Path to the model directory."},
    )
    dataset_path: Optional[str] = field(
        default="./dataset.npy",
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
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
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
        default=True,
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

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True
    save_strategy = "steps"


    # model
    model = create_and_prepare_model(script_args, training_args)
    model.config.use_cache = False

    # datasets
    ## ToDo uncomment once drive goes public
    # train_url = "https://drive.google.com/file/d/1-JgY1mEafcJ7qhggt6UR3OEKAciIPd5s/view?usp=sharing"
    # eval_url =  "https://drive.google.com/file/d/1jrm6Lacrq49AYv0uB_Qy22xRmfPixQvs/view?usp=sharing"
    # dataset = load_dataset("parquet", data_files={'train': train_url, 'validation': eval_url})
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{script_args.dataset_path}/train-00000-of-00001.parquet",
            "validation": f"{script_args.dataset_path}/validation-00000-of-00001.parquet",
        },
    )
    train_dataset, eval_dataset = dataset["train"], dataset["validation"]

    # trainer
    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MLPerfCallback(loralogger, len(train_dataset), len(eval_dataset),args.lora_alpha)],
    )
    trainer.accelerator.print(f"{trainer.model}")
    if script_args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if script_args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, training_args)

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GaudiTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args)
