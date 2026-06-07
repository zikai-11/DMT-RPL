import transformers

import modeling_textattack
import csv

from location_accuracy import NLI_infer_BERT


def get_my_data(path):
    text2 = []
    text1 = []
    label=[]
    with open(path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            label1 = row[2]
            tex = row[1]
            text = row[0]
            text1.append(text)
            text2.append(tex)
            label.append(label1)
    return text1,text2,label
#
text1,text2,label = get_my_data("data/attack/location_acc_data/bertattack_location_mr.tsv")

text1 = text1[:100]
text2 = text2[:100]
label = label[:100]
model = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/MR-adv",
                                                                            num_labels=2).cuda()


tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/MR-adv")

model_wrapper = NLI_infer_BERT(model,tokenizer=tokenizer,max_seq_length=128,num_classes=2)


freq,important,a = model_wrapper(text2,text1)

print(freq,important,a)