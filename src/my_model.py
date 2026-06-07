import re

import textattack
import transformers
from mymodel import NLI_infer_BERT
import modeling_textattack

import csv

from hhh import HuggingFaceModelWrapper1


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

# model = transformers.AutoModelForSequenceClassification.from_pretrained("experiments/MR-adv")

tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/ag_news-adv")
# model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
# model_wrapper = HuggingFaceModelWrapper1(model, tokenizer)

model_wrapper = NLI_infer_BERT(model,tokenizer=tokenizer,max_seq_length=128,num_classes=4)

attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
#dataset = textattack.datasets.HuggingFaceDataset("rotten_tomatoes", split="test")
my_dataset = textattack.datasets.Dataset(data)
# my_dataset = textattack.datasets.Dataset(data,input_columns=("premise", "hypothesis"))
# Attack 20 samples with CSV logging and  checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
                    num_examples = 100,
                    log_to_csv = "1.csv",
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



