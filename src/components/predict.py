from datasets import load_dataset, Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline
from dataclasses import dataclass
import os
import torch
import pandas as pd
import numpy as np
from src.components.Youtube_Scrapper import Get_Youtube_comments
from src.components.training import TrainingConfig, Model_Training
import torch
from src.exception import CustomException
from src.logger import logging
import sys

model = AutoModelForSequenceClassification.from_pretrained("artifacts/model", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("artifacts/tokenizer", return_tensors="pt")
#pipeline = pipeline("artifact/model", num_labels=2)

@dataclass
class PredictionConfig:
    paths = TrainingConfig
    model_path : str = paths.model_path
    tokenizer_path : str = paths.tokenizer_path

class Predictor:
    def __init__(self):
        self.predictionconfig = PredictionConfig()
        #self.id = id

    def predict(self, df):
        try:
            #tokenize=Model_Training.tokenize(df)
            id2label = {0:"negative", 1:"positive"}

            # comments = Get_Youtube_comments()
            # df = comments.IngestData(self.id)

            dataset = Dataset.from_pandas(df)
            dataset = dataset.remove_columns(['author', 'published_at', 'updated_at', 'like_count'])

            tokenized_data = dataset.map(Model_Training().tokenize, batched=True)

            #model = AutoModelForSequenceClassification(self.predictionconfig.model_path, num_labels=2)
            input_ids = torch.tensor(tokenized_data["input_ids"])
            attention_mask = torch.tensor(tokenized_data["attention_mask"])

            with torch.no_grad():
                predictions = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = predictions.logits

            Sigmoid = torch.nn.Sigmoid()

            probs = Sigmoid(preds)

            predictions = np.zeros(probs.cpu().numpy().shape)
            predictions[np.where(probs >= 0.5)] = 1

            predictions = np.argmax(predictions, axis=1)

            label = [id2label[pred] for pred in predictions]

            positive_pct = np.round(label.count("positive") / len(label) * 100, 2)
            negative_pct = np.round(label.count("negative") / len(label) * 100, 2)

            df["Sentiment"] = label

            return df
        except Exception as e:
            raise CustomException(e, sys)

