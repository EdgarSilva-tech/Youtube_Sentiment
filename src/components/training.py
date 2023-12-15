from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from dataclasses import dataclass
import os
from src.utils import compute_metrics
from src.logger import logging

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

@dataclass
class TrainingConfig:
    model_path : str = os.path.join("artifacts", "model")
    tokenizer_path : str = os.path.join("artifacts", "tokenizer")

class Model_Training:
    def __init__(self):
        self.trainconfig = TrainingConfig()

    def tokenize(self, data):
        return tokenizer(data["text"], truncation=True, padding="max_length")
    
    def preprocess(self, train_set):
        train_set.set_format("pandas")
        df_train_set = train_set[:]
        df_train_set["labels"] = df_train_set["labels"].replace(4,1)
        train_set = Dataset.from_pandas(df_train_set)
        return train_set

    def train(self, train, eval):

        train = load_from_disk(train)
        eval = load_from_disk(eval)

        tokenized_train=train.map(self.tokenize, batched=True)
        tokenized_eval=eval.map(self.tokenize, batched=True)

        train_set = self.preprocess(tokenized_train)
        eval_set = self.preprocess(tokenized_eval)

        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_args = TrainingArguments(
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            output_dir=self.trainconfig.model_path)

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        trainer.evaluate()

        logging.info("Model has been trained and evaluated")
        trainer.save_model(self.trainconfig.model_path)
        tokenizer.save_pretrained(self.trainconfig.tokenizer_path)

        return trainer, tokenizer