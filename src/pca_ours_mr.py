import csv
from umap import UMAP
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from scipy.stats import wasserstein_distance
import modeling_textattack
from pca_ours import ag_pca

# model = transformers.BertForSequenceClassification.from_pretrained("experiments/MR-base")
MLM = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/MR-adv",num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/MR-adv")

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import jensenshannon, cosine

# 1. MMD 距离（论文最认可，越小越好）
def mmd_linear(X, Y):
    """线性核MMD，低维空间最稳定，区分度最高"""
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()
def wasserstein_2d(X, Y):
    """
    计算二维特征的Wasserstein距离
    分别计算x轴和y轴，再取平均
    """
    wd_x = wasserstein_distance(X[:, 0], Y[:, 0])
    wd_y = wasserstein_distance(X[:, 1], Y[:, 1])

    return (wd_x + wd_y) / 2

def jsd_2d(X, Y, bins=20):
    """计算2D特征的JSD，保留空间分布信息"""
    # 计算联合范围，保证两个分布的直方图在同一坐标系下
    x_min = min(X[:, 0].min(), Y[:, 0].min())
    x_max = max(X[:, 0].max(), Y[:, 0].max())
    y_min = min(X[:, 1].min(), Y[:, 1].min())
    y_max = max(X[:, 1].max(), Y[:, 1].max())

    # 计算二维直方图
    hist_X, _ = np.histogramdd(X, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True)
    hist_Y, _ = np.histogramdd(Y, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True)

    # 归一化
    hist_X = hist_X.flatten() + 1e-8
    hist_Y = hist_Y.flatten() + 1e-8
    hist_X /= hist_X.sum()
    hist_Y /= hist_Y.sum()

    # 计算JSD
    M = 0.5 * (hist_X + hist_Y)
    return 0.5 * (np.sum(hist_X * np.log(hist_X / M)) + np.sum(hist_Y * np.log(hist_Y / M)))


def cos_sim_avg(X, Y):
    """计算两个分布中所有样本对的余弦相似度的平均值"""
    sim_matrix = cosine_similarity(X, Y)
    return sim_matrix.mean()


def euclidean_dist_avg(X, Y):
    """计算两个分布中所有样本对的欧氏距离的平均值，最直观"""
    dist_matrix = np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=2))
    return dist_matrix.mean()

def get_VAT_embeddeding(original,labels,batch_size=32):
    VAT_embed=[]
    labels_int = list(map(int, labels))
    labels = torch.tensor(labels_int, dtype=torch.long)
    for i in range(0, len(original),batch_size):
        batch = original[i:i+batch_size]
        labels_batch = torch.tensor(labels[i:i+batch_size])
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300  # BERT最大输入长度
        )
        input_ids = inputs["input_ids"]
        ##
        # mask_ids, output_labels = get_masked(input_ids)
        ##

        embeds_init = MLM.bert.embeddings.word_embeddings(input_ids) #换成input_ids
        tr_loss = 0
        adv_steps = 2
        adv_init_mag = 5e-2  # 1e-1
        adv_max_norm = 1  # 2e-1
        adv_lr = 3e-2  ## 1e-2
        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)  # 随机初始化一个和
        dims = torch.tensor(768, device=delta.device).float()
        mag = adv_init_mag / torch.sqrt(dims)  # B
        bs, seq_len = input_ids.size()
        delta = delta * mag.view(-1, 1, 1)

        for astep in range(adv_steps):
            delta.requires_grad_()
            inputs_embeds = embeds_init + delta
            outputs = MLM.forward_infer(
            inputs_embeds=inputs_embeds,inference=True
            )
            output = outputs[0]
            loss = 0
            loss_fct1 = CrossEntropyLoss()

            loss = loss_fct1(output.view(-1, 2), labels_batch.view(-1))

            if astep == adv_steps - 1:
                break

            loss.backward()
            # get grad on delta
            delta_grad = delta.grad.clone().detach()

            # clip

            # grad-norm
            denorm = torch.norm(delta_grad, dim=-1).view(bs, seq_len, 1)  # B seq-len 1
            denorm = torch.clamp(denorm, min=1e-8)
            # add the delta with grads
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # B seq-len D

            # normalize new delta at token-level
            delta_norm = torch.norm(delta, p=2, dim=-1).detach()  # B seq-len
            mean_norm, _ = torch.max(delta_norm, dim=-1, keepdim=True)  # B,1
            # reweight-delta using scaling
            reweights_tok = (delta_norm / mean_norm).view(bs, seq_len, 1)  # B seq-len, 1
            delta = delta * reweights_tok

            # reweight the exceed delta
            delta_norm = torch.norm(delta.view(bs, -1).float(), p=2, dim=1).detach()
            exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
            reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)  # B 1 1

            # detach delta and embeds init for next iteration
            delta = (delta * reweights).detach()
            embeds_init = MLM.bert.embeddings.word_embeddings(input_ids) #换成input_ids
        with torch.no_grad():
            outputs_bert = MLM.bert(inputs_embeds=inputs_embeds) ##model.bert
            ##
            # mask_output = MLM.cls(outputs_bert[0])
            # adv_ids = torch.argmax(mask_output,dim=-1)
            # outputs_bert = model.bert(input_ids=adv_ids)
            ##

        batch_embeddings = outputs_bert.last_hidden_state[:, 0, :].numpy()
        VAT_embed.extend(batch_embeddings)
    return np.array(VAT_embed)

def get_ours_embeddeding(original,labels,batch_size=32):
    VAT_embed=[]
    labels_int = list(map(int, labels))
    labels = torch.tensor(labels_int, dtype=torch.long)
    for i in range(0, len(original),batch_size):
        batch = original[i:i+batch_size]
        labels_batch = torch.tensor(labels[i:i+batch_size])
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300  # BERT最大输入长度
        )
        input_ids = inputs["input_ids"]
        ##
        mask_ids, output_labels = get_masked(input_ids)
        ##

        embeds_init = MLM.bert.embeddings.word_embeddings(mask_ids) #换成input_ids
        tr_loss = 0
        adv_steps = 2
        adv_init_mag = 5e-2  # 1e-1
        adv_max_norm = 1  # 2e-1
        adv_lr = 3e-2  ## 1e-2
        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)  # 随机初始化一个和
        dims = torch.tensor(768, device=delta.device).float()
        mag = adv_init_mag / torch.sqrt(dims)  # B
        bs, seq_len = input_ids.size()
        delta = delta * mag.view(-1, 1, 1)

        for astep in range(adv_steps):
            delta.requires_grad_()
            inputs_embeds = embeds_init + delta
            outputs = MLM.forward_infer(
            inputs_embeds=inputs_embeds
            ,inference=True)
            output = outputs[0]
            loss = 0
            loss_fct1 = CrossEntropyLoss()

            loss1 = loss_fct1(output.view(-1, 2), labels_batch.view(-1))
            loss2 = loss_fct1(outputs[1].view(-1, 30522), output_labels.view(-1))
            loss = loss1 + loss2
            if astep == adv_steps - 1:
                break

            loss.backward()
            # get grad on delta
            delta_grad = delta.grad.clone().detach()

            # clip

            # grad-norm
            denorm = torch.norm(delta_grad, dim=-1).view(bs, seq_len, 1)  # B seq-len 1
            denorm = torch.clamp(denorm, min=1e-8)
            # add the delta with grads
            delta = (delta + adv_lr * delta_grad / denorm).detach()  # B seq-len D

            # normalize new delta at token-level
            delta_norm = torch.norm(delta, p=2, dim=-1).detach()  # B seq-len
            mean_norm, _ = torch.max(delta_norm, dim=-1, keepdim=True)  # B,1
            # reweight-delta using scaling
            reweights_tok = (delta_norm / mean_norm).view(bs, seq_len, 1)  # B seq-len, 1
            delta = delta * reweights_tok

            # reweight the exceed delta
            delta_norm = torch.norm(delta.view(bs, -1).float(), p=2, dim=1).detach()
            exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
            reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)  # B 1 1

            # detach delta and embeds init for next iteration
            delta = (delta * reweights).detach()
            embeds_init = MLM.bert.embeddings.word_embeddings(mask_ids) #换成input_ids
        with torch.no_grad():
            outputs_bert = MLM.bert(inputs_embeds=inputs_embeds) ##model.bert
            ##
            mask_output = MLM.cls(outputs_bert[0])
            adv_ids = torch.argmax(mask_output,dim=-1)
            # for i in range(adv_ids.shape[0]):
            #     seten = tokenizer.decode(adv_ids[i].cpu().numpy(),skip_special_tokens=True)
            #     print(seten)
            outputs_bert = MLM.bert(input_ids=adv_ids)
            ##

        batch_embeddings = outputs_bert.last_hidden_state[:, 0, :].numpy()
        VAT_embed.extend(batch_embeddings)
    return np.array(VAT_embed)



def get_masked(input_ids, mlm_probability=0.15):
    output_labels = input_ids.clone()
    input_ids = input_ids.clone()
    #output_labels[output_labels == pad_token_id] = -100
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    # special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
    #                          for val in
    #                        input_ids.tolist()]
    #
    # special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    # probability_matrix = probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    output_labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(output_labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = 103  # hard code mask index

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(output_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # hard code random word
    random_words = torch.randint(30522, output_labels.shape, dtype=torch.long, device=output_labels.device)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, output_labels

def get_my_data(path):
    data = []
    data1 = []
    labels = []
    with open(path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue  # 跳过第一行
            text1 = row[1]
            text = row[0]
            label = row[2]
            data.append(text)
            data1.append(text1)
            labels.append(label)
    return data,data1,labels

def get_embeddings(data,batch_size=32):
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300  # BERT最大输入长度
        )
        with torch.no_grad():
            outputs = MLM.bert(**inputs)
        # 取[CLS]标记的最后一层隐藏状态作为句子嵌入
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

attack,original,labels = get_my_data("attack_base_data/ours/data.tsv")
original = original[:500]
labels = labels[:500]
embeddings = get_embeddings(attack)
embeddings_VAT=get_ours_embeddeding(original,labels=labels)
embeddings_VAT_1=get_VAT_embeddeding(original,labels=labels)

embeddings_ag,embeddings_VAT_ag,embeddings_VAT_1_ag = ag_pca()

# result = np.vstack([
#     np.vstack([embeddings[i:i+500], embeddings_ag[i:i+500]])
#     for i in range(0, 2000, 500)
# ])
##定量计算
textfooler   = np.vstack([embeddings[0:500], embeddings_ag[0:500]])
pwws         = np.vstack([embeddings[500:1000], embeddings_ag[500:1000]])
bert_attack  = np.vstack([embeddings[1000:1500], embeddings_ag[1000:1500]])
deepwordbug  = np.vstack([embeddings[1500:2000], embeddings_ag[1500:2000]])

# 拼接Ours和VAT
ours_total = np.vstack([embeddings_VAT, embeddings_VAT_ag])
vat_total  = np.vstack([embeddings_VAT_1, embeddings_VAT_1_ag])

# ========== 3. 关键修正：在所有样本上拟合PCA ==========
# 把所有参与对比的样本合并，训练PCA，保证投影空间公平
all_feat = np.vstack([textfooler, pwws, bert_attack, deepwordbug, ours_total, vat_total])
pca = PCA(n_components=2)
all_feat_2d = pca.fit_transform(all_feat)

# 按切片拆分，还原各组2D PCA坐标
len_tf = textfooler.shape[0]
len_pw = pwws.shape[0]
len_ba = bert_attack.shape[0]
len_db = deepwordbug.shape[0]
len_ours = ours_total.shape[0]
len_vat = vat_total.shape[0]

tf_2d = all_feat_2d[:len_tf]
pw_2d = all_feat_2d[len_tf : len_tf+len_pw]
ba_2d = all_feat_2d[len_tf+len_pw : len_tf+len_pw+len_ba]
db_2d = all_feat_2d[len_tf+len_pw+len_ba : len_tf+len_pw+len_ba+len_db]
ours_2d = all_feat_2d[len_tf+len_pw+len_ba+len_db : len_tf+len_pw+len_ba+len_db+len_ours]
vat_2d = all_feat_2d[len_tf+len_pw+len_ba+len_db+len_ours : ]

attack_2d_list = [
    ("TextFooler", tf_2d),
    ("PWWS", pw_2d),
    ("BERT-Attack", ba_2d),
    ("DeepWordBug", db_2d)
]

# ========== 4. 输出结果（和你的PCA图100%匹配） ==========
print("\n" + "="*80)
print("✅ Ours (PCA 2D特征) → 四种攻击")
print("="*80)
for name, feat2d in attack_2d_list:
    mmd = mmd_linear(ours_2d, feat2d)
    jsd = jsd_2d(ours_2d, feat2d)
    wd = wasserstein_2d(ours_2d, feat2d)
    euc = euclidean_dist_avg(ours_2d, feat2d)
    print(f"Ours → {name:12s} | MMD={mmd:.6f} | JSD={jsd:.4f} | WD={wd:.4f} | Euc={euc:.4f}")

print("\n" + "="*80)
print("✅ VAT (PCA 2D特征) → 四种攻击")
print("="*80)
for name, feat2d in attack_2d_list:
    mmd = mmd_linear(vat_2d, feat2d)
    jsd = jsd_2d(vat_2d, feat2d)
    wd = wasserstein_2d(vat_2d, feat2d)
    euc = euclidean_dist_avg(vat_2d, feat2d)
    print(f"VAT  → {name:12s} | MMD={mmd:.6f} | JSD={jsd:.4f} | WD={wd:.4f} | Euc={euc:.4f}")
# labels_1 = np.repeat([0, 1, 2, 3 ,4, 5], repeats=1000)
# all_embeddings = np.vstack([result, embeddings_VAT_1, embeddings_VAT_ag, embeddings_VAT, embeddings_VAT_1_ag])
#
# # 标准化数据（UMAP对尺度敏感，建议保留标准化）
# scaler = StandardScaler()
# embeddings_scaled = scaler.fit_transform(all_embeddings)
#
# # 修改部分：替换PCA为UMAP
# umap_model = UMAP(
#     n_components=2,
#     n_neighbors=150,  # 增大邻域范围以捕捉全局重叠
#     min_dist=0.3,  # 放松点间距，避免过度压缩重叠区域
#     metric='cosine',  # 适合文本/图像嵌入的对抗样本分析
#     repulsion_strength=1.5,  # 增强类间排斥以暴露重叠
#     spread=1.2,  # 控制分布范围
#     random_state=42,
#     local_connectivity=2  # 提升局部结构敏感度
# )
# embeddings_umap = umap_model.fit_transform(embeddings_scaled)
#
# # 创建新的DataFrame
# data = pd.DataFrame({
#     "umap_x": embeddings_umap[:, 0],  # 修改列名
#     "umap_y": embeddings_umap[:, 1],  # 修改列名
#     "label": labels_1
# })
#
# # 定义颜色和类别名称（保持不变）
# colors = ["red", "blue", "green", "purple","yellow" ,"orange"]
# class_names = ["TextFooler", "PWWS", "BERT-Attack", "DeepWordBug", "VAT", "Ours"]
#
# # 绘制UMAP结果
# plt.figure(figsize=(10, 8))  # 建议稍大画布
# for i in range(6):
#     mask = data["label"] == i
#     plt.scatter(
#         data.loc[mask, "umap_x"],
#         data.loc[mask, "umap_y"],
#         c=colors[i],
#         label=class_names[i],
#         alpha=0.6,
#         s=15          # 调整点大小
#     )
#
# # 添加图例和样式优化
# plt.legend()
# plt.grid(True, alpha=0.3)    # 降低网格线透明度
# plt.gca().set_aspect('auto')  # 自动调整纵横比
#
# plt.savefig("umap_embed.png", dpi=500)  # 保持完整布局
# plt.show()



# labels_1 = np.repeat([0, 1, 2, 3 ,4, 5], repeats=1000)
#
# all_embeddings = np.vstack([result, embeddings_VAT_1,embeddings_VAT_ag,embeddings_VAT,embeddings_VAT_1_ag])  # 假设 embeddings 和 embeddings_VAT 形状一致
#
# # 标准化数据
# scaler = StandardScaler()
# embeddings_scaled = scaler.fit_transform(all_embeddings)
#
# # PCA降维
# pca = PCA(n_components=2)
# embeddings_pca = pca.fit_transform(embeddings_scaled)
#
# # 创建 DataFrame 保存结果
# data = pd.DataFrame({
#     "pca_x": embeddings_pca[:, 0],  # PCA第一主成分
#     "pca_y": embeddings_pca[:, 1],  # PCA第二主成分
#     "label": labels_1  # 标签
# })
#
# # 定义颜色和类别名称
# colors = ["red", "blue", "green", "purple","orange" ,"yellow"]
# class_names = ["TextFooler", "PWWS", "BERT-Attack", "DeepWordBug", "TAVAT", "Ours"]
#
# # 绘制PCA结果
# plt.figure(figsize=(10, 8))
# for i in range(6):
#     # 筛选当前类别的数据
#     mask = data["label"] == i
#     plt.scatter(
#         data.loc[mask, "pca_x"],
#         data.loc[mask, "pca_y"],
#         c=colors[i],
#         label=class_names[i],
#         alpha=0.6
#     )
#
# # 添加标签和标题
# # plt.xlabel("PCA Component 1")
# # plt.ylabel("PCA Component 2")
# plt.tick_params(axis='both', which='major', labelsize=20)  # 调整主刻度标签大小
# # 调整图例字体大小
# plt.legend(prop={'size': 15},markerscale=1.5)  # 显示图例并设置字体大小
#
# plt.grid(True)
#
# plt.savefig("embed.pdf",dpi=500)
#
# plt.show()


