# from transformers import TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_dataset, load_from_disk
# from textSummarizer.entity import ModelTrainerConfig
# import torch
# import os


# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config


    
#     def train(self):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#         model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
#         seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
#         #loading data 
#         dataset_samsum_pt = load_from_disk(self.config.data_path)

#         # trainer_args = TrainingArguments(
#         #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
#         #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
#         #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
#         #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
#         #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
#         # ) 


#         trainer_args = TrainingArguments(
#             output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
#             per_device_train_batch_size=1, per_device_eval_batch_size=1,
#             weight_decay=0.01, logging_steps=10,
#             evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
#             gradient_accumulation_steps=16
#         ) 

#         trainer = Trainer(model=model_pegasus, args=trainer_args,
#                   tokenizer=tokenizer, data_collator=seq2seq_data_collator,
#                   train_dataset=dataset_samsum_pt["train"], 
#                   eval_dataset=dataset_samsum_pt["validation"])
        
#         trainer.train()

#         ## Save model
#         model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
#         ## Save tokenizer
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))

import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
from torch import device
from accelerate import DataLoaderConfiguration

from textSummarizer.entity import ModelTrainerConfig



class ModelTrainer:
    # def __init__(self, config: ModelTrainerConfig):
    #     self.config = config
    def __init__(self, config: ModelTrainerConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer  # Store tokenizer in the class




    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        # model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        # seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # **Fine-tuned BART Support (replace with your fine-tuned model and tokenizer paths)**
        model_ckpt = "facebook/bart-base"  # Replace with actual path
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)

        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # )

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        )

        # trainer = Trainer(model=model_pegasus, args=trainer_args,
        #           tokenizer=tokenizer, data_collator=seq2seq_data_collator,
        #           train_dataset=dataset_samsum_pt["train"], 
        #           eval_dataset=dataset_samsum_pt["validation"])
        
        # trainer.train()

        # ## Save model
        # model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        # ## Save tokenizer
        # tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
        
        trainer = Trainer(
            model=model_bart,  # Use the fine-tuned BART model here
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],#test tacking log time to train data    
                eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model and tokenizer
        model_bart.save_pretrained(os.path.join(self.config.root_dir, "bart-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))