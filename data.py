from datasets import Dataset
import json

def get_dataset():
    data_list = []

    print(" --- Loading dataset...")
    with open("./dataset/train_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    return Dataset.from_list(data_list)
