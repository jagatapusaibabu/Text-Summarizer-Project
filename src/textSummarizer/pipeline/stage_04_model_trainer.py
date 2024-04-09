
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logging import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # config = ConfigurationManager()
        # model_trainer_config = config.get_model_trainer_config()
        # model_trainer_config = ModelTrainer(config=model_trainer_config)
        # model_trainer_config.train()
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        # **Define tokenizer outside**
        model_ckpt = "facebook/bart-base"  # Replace with actual path
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model_trainer = ModelTrainer(config=model_trainer_config, tokenizer=tokenizer)
        model_trainer.train()
