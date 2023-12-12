from src.logger import logging
from src.exception import CustomException
from src.components.training import Model_Training
import os
import sys
from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts", "train_data")
    test_data_path : str = os.path.join("artifacts", "test_data")
    raw_data_path : str = os.path.join("artifacts", "raw_data")

class Ingest_Data:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def get_data(self):
        try:
            logging.info("Loading the dataset")
            tweets = load_dataset("sentiment140")
            tweets.save_to_disk(self.ingestion_config.raw_data_path)

            tweets = tweets.remove_columns(["date", "user", "query"])
            tweets = tweets.rename_column("sentiment", "labels")

            logging.info("Initiating the dataset split")
            split = tweets["train"].train_test_split(test_size=0.1)
            train_set = split["train"]
            eval_set = split["test"]

            train_set = train_set.shuffle(seed=42).select([i for i in list(range(10000))])
            eval_set = eval_set.shuffle(seed=42).select([i for i in list(range(100))])

            train_set.save_to_disk(self.ingestion_config.train_data_path)

            eval_set.save_to_disk(self.ingestion_config.test_data_path)

            logging.info("Data Ingestion has been completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=Ingest_Data()
    train_data, test_data=obj.get_data()

    train = Model_Training()
    trainer, tokenizer = train.train(train_data, test_data)

