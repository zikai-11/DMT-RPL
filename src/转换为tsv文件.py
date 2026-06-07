import os

from tqdm import tqdm
import pandas

labels = []
sentences = []
with open("data/sst/train") as f:
    lines = f.readlines()
lines= lines[:80000]
i = 0
for line in tqdm(lines):
    line = line.strip()
    line = line.replace('\n ', ' ').replace('\t', ' ')
    if line == '':
        continue
    if line[:1] == '0':
        labels.append(0)
        sentences.append(line.replace('0', ''))
    if line[:1] == '1':
        labels.append(1)
        sentences.append(line.replace('1', ''))
    if line[:1] == '2':
        labels.append(2)
        sentences.append(line.replace('2', ''))
    if line[:1] == '3':
        labels.append(3)
        sentences.append(line.replace('3', ''))
    i += 1
result_dataframe = pandas.DataFrame({'sentence': sentences, 'label': labels})#'ID':
result_dataframe.to_csv(os.path.join("data/ag_news/original_data/", 'train.tsv'), sep='\t', index=False)