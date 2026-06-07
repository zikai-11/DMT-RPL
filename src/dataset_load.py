import random

import datasets
import os
def dataset_mapping(x):
        return {
            "x": x["sentence"],
            "y": 1 if x["label"] > 0.5 else 0,
        }
dataset =datasets.load_dataset("stanfordnlp/sst2", split="train",cache_dir="data")
# print(dataset)
# print(dataset[0])
texts=[]
labels=[]
for item in dataset:
    texts.append(item["text"])
    labels.append(item["label"])
combined = list(zip(texts, labels))
random.shuffle(combined)
texts_shuffled, labels_shuffled = zip(*combined)

# 将打乱后的数据集写入文件
with open(os.path.join("data/sst/", 'train'), 'w') as ofile:
    for text, label in zip(texts_shuffled, labels_shuffled):
        ofile.write('{} {}\n'.format(label, text))