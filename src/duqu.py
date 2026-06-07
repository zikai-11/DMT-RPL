import csv
import os

import pandas

labels = []
attack_text = []
original_text = []
with open("attack_base_data/ours/bertattack-ours-500.csv", 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for i,row in enumerate(reader):
        if i==0 :
            continue
        if row[8] != "Skipped":
            attack_text1 = row[1]
            original_text1 = row[0]
            attack_text1 = attack_text1.replace("[[", "").replace("]]", "")
            label1 = row[6]
            # labels1 = labels1.replace("[[", "").replace("]]", "")
            original_text1 = original_text1.replace("[[", "").replace("]]", "")
            attack_text.append(attack_text1)
            labels.append(label1)
            original_text.append(original_text1)

attack_text = attack_text[:]
# labels = labels[:]


result_dataframe = pandas.DataFrame({'attack': attack_text,"original":original_text,"label":labels})#'ID':
result_dataframe.to_csv(os.path.join("data/attack/location_acc_data/", 'bertattack_location_mr.tsv'), sep='\t', index=False)
