import numpy as np
from datasets import load_metric
import evaluate
import os
import pickle
from src.exception import CustomException
import sys

def compute_metrics(eval_pred):
    load_acc = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = load_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=preds, references=labels)["f1"]

    return {"accuracy": acc, "f1": f1}

def save_model(model_path, obj):
    try:
        dir_path = os.dirname(model_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(model_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)