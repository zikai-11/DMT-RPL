import pandas as pd
import seaborn as sns
import torch
import transformers

import numpy as np


from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity


import modeling_textattack
import csv

from embedding_experimenst import NLI_infer_BERT_mr,NLI_infer_BERT_ag,NLI_infer_BERT_imdb



def embedding(text1,text2,tokenizer,model_wrapper,model):
    inputs_dict_attack = tokenizer(
        text1,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    embedding_pur = model_wrapper(text1)

    input_ids1 = inputs_dict_attack["input_ids"].clone().detach().to(device='cuda')
    input_mask1 = inputs_dict_attack["attention_mask"].clone().detach().to(device='cuda')
    segment_ids1 = inputs_dict_attack["token_type_ids"].clone().detach().to(device='cuda')

    inputs_dict_original = tokenizer(
        text2,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids2 = inputs_dict_original["input_ids"].clone().detach().to(device='cuda')
    input_mask2 = inputs_dict_original["attention_mask"].clone().detach().to(device='cuda')
    segment_ids2 = inputs_dict_original["token_type_ids"].clone().detach().to(device='cuda')
    pur_embedding = embedding_pur
    attack_embedding = model.bert(input_ids1, input_mask1, segment_ids1)[1]
    # attack_embedding = torch.argmax(attack_embedding,dim=-1)
    original_embedding = model.bert(input_ids2, input_mask2, segment_ids2)[1]
    # original_embedding = torch.argmax(original_embedding,dim=-1)
    # pur_embedding = torch.argmax(pur_embedding,dim=-1)
    # bert.embeddings.word_embeddings
    original_embedding = original_embedding.reshape(original_embedding.shape[0], -1)
    pur_embedding = pur_embedding.reshape(original_embedding.shape[0], -1)
    attack_embedding = attack_embedding.reshape(original_embedding.shape[0], -1)

    # Assuming attack_embedding and original_embedding are tensors, move to CPU and convert to numpy arrays
    attack_embedding = attack_embedding.cpu().detach().numpy()
    pur_embedding = pur_embedding.cpu().detach().numpy()
    original_embedding = original_embedding.cpu().detach().numpy()

    similarity_scores_clean_to_reconstructed = cosine_similarity(original_embedding, pur_embedding)
    similarity_scores_clean_to_reconstructed = np.diag(similarity_scores_clean_to_reconstructed)

    # 计算干净样本和对抗性样本之间的余弦相似度
    similarity_scores_clean_to_adversarial = cosine_similarity(original_embedding, attack_embedding)
    similarity_scores_clean_to_adversarial = np.diag(similarity_scores_clean_to_adversarial)

    data_to_plot = [similarity_scores_clean_to_adversarial, similarity_scores_clean_to_reconstructed]

    return data_to_plot
def embedding_imdb(text1,text2,tokenizer,model_wrapper,model):
    inputs_dict_attack = tokenizer(
        text1,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    embedding_pur = model_wrapper(text1)

    input_ids1 = inputs_dict_attack["input_ids"].clone().detach().to(device='cuda:1')
    input_mask1 = inputs_dict_attack["attention_mask"].clone().detach().to(device='cuda:1')
    segment_ids1 = inputs_dict_attack["token_type_ids"].clone().detach().to(device='cuda:1')

    inputs_dict_original = tokenizer(
        text2,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids2 = inputs_dict_original["input_ids"].clone().detach().to(device='cuda:1')
    input_mask2 = inputs_dict_original["attention_mask"].clone().detach().to(device='cuda:1')
    segment_ids2 = inputs_dict_original["token_type_ids"].clone().detach().to(device='cuda:1')
    pur_embedding = embedding_pur
    attack_embedding = model.bert(input_ids1, input_mask1, segment_ids1)[1]
    # attack_embedding = torch.argmax(attack_embedding,dim=-1)
    original_embedding = model.bert(input_ids2, input_mask2, segment_ids2)[1]
    # original_embedding = torch.argmax(original_embedding,dim=-1)
    # pur_embedding = torch.argmax(pur_embedding,dim=-1)
    # bert.embeddings.word_embeddings
    original_embedding = original_embedding.reshape(original_embedding.shape[0], -1)
    pur_embedding = pur_embedding.reshape(original_embedding.shape[0], -1)
    attack_embedding = attack_embedding.reshape(original_embedding.shape[0], -1)

    # Assuming attack_embedding and original_embedding are tensors, move to CPU and convert to numpy arrays
    attack_embedding = attack_embedding.cpu().detach().numpy()
    pur_embedding = pur_embedding.cpu().detach().numpy()
    original_embedding = original_embedding.cpu().detach().numpy()

    similarity_scores_clean_to_reconstructed = cosine_similarity(original_embedding, pur_embedding)
    similarity_scores_clean_to_reconstructed = np.diag(similarity_scores_clean_to_reconstructed)

    # 计算干净样本和对抗性样本之间的余弦相似度
    similarity_scores_clean_to_adversarial = cosine_similarity(original_embedding, attack_embedding)
    similarity_scores_clean_to_adversarial = np.diag(similarity_scores_clean_to_adversarial)

    data_to_plot = [similarity_scores_clean_to_adversarial, similarity_scores_clean_to_reconstructed]

    return data_to_plot
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
text1_mr,text2_mr,label_mr = get_my_data("data/attack/mr.tsv")
text1_ag,text2_ag,label_ag = get_my_data("data/attack/ag.tsv")
text1_imdb,text2_imdb,label_imdb = get_my_data("data/attack/yelp.tsv")


text1_mr= text1_mr[:100]
text2_mr = text2_mr[:100]

text1_ag= text1_ag[:100]
text2_ag = text2_ag[:100]

text1_imdb= text1_imdb[:100]
text2_imdb = text2_imdb[:100]

model_mr = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/MR-adv",
                                                                            num_labels=2).cuda()

# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
#
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

model = transformers.AutoModelForSequenceClassification.from_pretrained("experiments/MR-base")

# tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/MR-base")

tokenizer_mr = transformers.AutoTokenizer.from_pretrained("experiments/MR-adv")

model_ag = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/ag_news-adv",
                                                                            num_labels=4).cuda()

tokenizer_ag = transformers.AutoTokenizer.from_pretrained("experiments/ag_news-adv")

model_imdb = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/yelp-adv",
                                                                            num_labels=2).to(device='cuda:1')

tokenizer_imdb = transformers.AutoTokenizer.from_pretrained("experiments/yelp-adv")

model_wrapper_mr = NLI_infer_BERT_mr(model_mr,tokenizer=tokenizer_mr,max_seq_length=128,num_classes=2)

model_wrapper_ag = NLI_infer_BERT_ag(model_ag,tokenizer=tokenizer_ag,max_seq_length=128,num_classes=4)

model_wrapper_imdb = NLI_infer_BERT_imdb(model_imdb,tokenizer=tokenizer_imdb,max_seq_length=128,num_classes=2)

data_to_plot_mr = embedding(text1_mr,text2_mr,tokenizer_mr,model_wrapper_mr,model_mr)

data_to_plot_ag = embedding(text1_ag,text2_ag,tokenizer_ag,model_wrapper_ag,model_ag)

data_to_plot_imdb = embedding_imdb(text1_imdb,text2_imdb,tokenizer_imdb,model_wrapper_imdb,model_imdb)
# 创建箱线图



# Convert lists to DataFrame
data_mr_df = pd.DataFrame({
    'Values': np.concatenate((np.atleast_1d(data_to_plot_mr[0]), np.atleast_1d(data_to_plot_mr[1]))),
    'Category': ['Adversarial and Clean']*len(data_to_plot_mr[0]) + ['Purified and Clean']*len(data_to_plot_mr[1]),
    'Dataset': ['MR']*len(data_to_plot_mr[0]) + ['MR']*len(data_to_plot_mr[1])
})

data_ag_df = pd.DataFrame({
    'Values': np.concatenate((np.atleast_1d(data_to_plot_ag[0]), np.atleast_1d(data_to_plot_ag[1]))),
    'Category': ['Adversarial and Clean']*len(data_to_plot_ag[0]) + ['Purified and Clean']*len(data_to_plot_ag[1]),
    'Dataset': ['AG’s News']*len(data_to_plot_ag[0]) + ['AG’s News']*len(data_to_plot_ag[1])
})

data_imdb_df = pd.DataFrame({
    'Values': np.concatenate((np.atleast_1d(data_to_plot_imdb[0]), np.atleast_1d(data_to_plot_imdb[1]))),
    'Category': ['Adversarial and Clean']*len(data_to_plot_imdb[0]) + ['Purified and Clean']*len(data_to_plot_imdb[1]),
    'Dataset': ['Yelp']*len(data_to_plot_imdb[0]) + ['Yelp']*len(data_to_plot_imdb[1])
})
data_mr_df['Group'] = data_mr_df['Dataset'] + " - " + data_mr_df['Category']
data_ag_df['Group'] = data_ag_df['Dataset'] + " - " + data_ag_df['Category']
data_imdb_df['Group'] = data_imdb_df['Dataset'] + " - " + data_imdb_df['Category']

# Combine all datasets
combined_data = pd.concat([data_mr_df, data_ag_df, data_imdb_df])
color_palette = {
    'Adversarial and Clean': '#F8766D',  # 匹配Gene1的红色
    'Purified and Clean': '#7CAE00'      # 匹配Gene3的绿色
}
# Plotting
plt.figure(figsize=(10, 6))

# 1. 绘制小提琴图（按Dataset分组，hue区分Category）
sns.violinplot(
    x='Dataset',        # x轴为“数据集”
    y='Values',         # y轴为“余弦相似度”
    hue='Category',     # 按“类别”区分小提琴
    data=combined_data,
    inner='box',        # 内部显示箱线图（四分位）
    linewidth=1.5,      # 小提琴轮廓线宽
    palette=color_palette,   # 配色（可替换为自定义调色板，如之前的color_palette）
    split=False,        # 不拆分小提琴（每个hue对应独立小提琴）
    scale='width'       # 小提琴宽度统一，便于对比
)

# 2. 叠加散点图（显示原始数据点）
sns.stripplot(
    x='Dataset',
    y='Values',
    hue='Category',
    data=combined_data,

    color='black',      # 散点颜色（统一黑色突出分布）
    size=3,             # 散点大小
    jitter=True,        # 抖动散点避免重叠
    alpha=0.5,          # 散点透明度
    dodge=True,         # 按hue分组偏移散点，避免重叠
    legend=False        # 散点不重复生成图例
)

# 3. 调整图例（位于下方中央）
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.3),  # 垂直位置更靠下，减少重叠
    ncol=2,
    fontsize=20  # 可根据需求调整字体大小，平衡显示与空间
)

# 调整画布布局，下方预留更多空间给图例
plt.tight_layout(rect=[0, 0.15, 1, 1])  # rect=[左, 下, 右, 上]，下方从0.15开始

# 4. 轴标签与字体
plt.xlabel(xlabel=None)  # 若需x轴标签，可根据需求添加（如“Dataset”）
plt.ylabel('Cosine Similarity', fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# # 5. 调整布局（给下方图例预留空间）
# plt.tight_layout(rect=[0, 0.1, 1, 1])  # rect=[左, 下, 右, 上]，下方从0.1开始，避免图例重叠

# 6. 保存并显示
plt.savefig(
    "cos_legend_below.pdf",
    dpi=500,
    format='pdf',
    bbox_inches='tight'  # 确保保存时包含所有元素（含下方图例）
)
plt.show()

#plt.title('Violin Plot Comparison (Adversarial vs Purified)', fontsize=16)  # 添加标题并设置字体大小
#plt.xlabel('Dataset', fontsize=20)  # 设置 x 轴标签并设置字体大小

# plt.figure(figsize=(10, 8), dpi=300)
#
#
# boxprops = dict(linewidth=2, color='blue')  # 箱体属性设置
# medianprops = dict(linestyle='-', linewidth=2.5, color='red')  # 中位数线属性设置
# whiskerprops = dict(linestyle='--', linewidth=1.5, color='black')  # 须线属性设置
# capprops = dict(linewidth=1.5, color='black')  # 端点线属性设置
# colors = ['blue', 'blue']
#
# bp_mr = plt.boxplot(data_to_plot_mr, patch_artist=True,
#                     boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
#
# for box, color in zip(bp_mr['boxes'], colors):
#     box.set(color=color, facecolor='w')  # 设置箱体的边框颜色和填充色
#
# plt.ylabel('Cosine Similarity', fontsize=15)
# plt.grid(False)
# plt.xticks([1, 2], ['Adversarial', 'Reconstructed'], fontsize=12)
# plt.tick_params(axis='y', labelsize=12)
# plt.title('Data Set MR', fontsize=15)
#
#
# bp_ag = plt.boxplot(data_to_plot_ag, patch_artist=True,
#                     boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
#
# for box, color in zip(bp_ag['boxes'], colors):
#     box.set(color=color, facecolor='w')  # 设置箱体的边框颜色和填充色
#
# plt.ylabel('Cosine Similarity', fontsize=15)
# plt.grid(False)
# plt.xticks([1, 2], ['Adversarial', 'Reconstructed'], fontsize=12)
# plt.tick_params(axis='y', labelsize=12)
# plt.title('Data Set AG', fontsize=15)
#
#
# bp_imdb = plt.boxplot(data_to_plot_imdb, patch_artist=True,
#                       boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
# for box, color in zip(bp_imdb['boxes'], colors):
#     box.set(color=color, facecolor='w')  # 设置箱体的边框颜色和填充色
#
# plt.ylabel('Cosine Similarity', fontsize=15)
# plt.grid(False)
# plt.xticks([1, 2], ['Adversarial', 'Reconstructed'], fontsize=12)
# plt.tick_params(axis='y', labelsize=12)
# plt.title('Data Set Yelp', fontsize=15)
# #
# plt.tight_layout()
# plt.show()

# 设置箱线图的颜色和样式




