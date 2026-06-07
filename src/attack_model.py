import re

import textattack
import transformers
from mymodel import NLI_infer_BERT
# from hhh import HuggingFaceModelWrapper1

import modeling_textattack

# def clean_str(string, TREC=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Every dataset is lower cased except for TREC
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip() if TREC else string.strip().lower()


import csv
# def get_my_data(path):
#     data = []
#     with open(path, 'r', encoding='utf-8') as fin:
#         reader = csv.reader(fin, delimiter='\t')
#         for idx, row in enumerate(reader):
#             if idx == 0:
#                 continue  # 跳过第一行
#             label = int(row[2])
#             text1 = row[0]
#             text2 = row[1]
#             data.append(((text1,text2), label))
#     return data
def get_my_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue  # 跳过第一行
            label = int(row[1])
            text = row[0]
            data.append((text, label))
    return data
# #
data = get_my_data("data/ag_news/original_data/test.tsv")



model = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/ag_news-adv",
                                                                            num_labels=4).cuda()



tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/ag_news-adv")

# model_wrapper = HuggingFaceModelWrapper1(model, tokenizer)

model_wrapper = NLI_infer_BERT(model,tokenizer=tokenizer,max_seq_length=128,num_classes=4)

attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
#dataset = textattack.datasets.HuggingFaceDataset("rotten_tomatoes", split="test")
my_dataset = textattack.datasets.Dataset(data)
# my_dataset = textattack.datasets.Dataset(data,input_columns=("premise", "hypothesis"))
# Attack 20 samples with CSV logging and  checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
                    num_examples = 300,
                    log_to_csv = "log_agpwws.csv",
                    checkpoint_interval = 1000,
                    disable_stdout = False,
                    log_summary_to_json="attack/imdb/default/"
                    )
# num_examples = 1000,
# log_to_csv = "log_IMDB_PWWS.csv",
# checkpoint_interval = 1000,
# checkpoint_dir = "attack/ag_news/default/",
# disable_stdout = False,
# log_summary_to_json = "attack/ag_news/default/"
attacker = textattack.Attacker(attack, my_dataset, attack_args)
results=attacker.attack_dataset()



