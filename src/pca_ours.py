import csv

import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

import modeling_textattack


# model = transformers.BertForSequenceClassification.from_pretrained("experiments/MR-base")
MLM = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/yelp-adv",num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained("experiments/yelp-adv")
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
            for i in range(adv_ids.shape[0]):
                seten = tokenizer.decode(adv_ids[i].cpu().numpy(),skip_special_tokens=True)
                print(seten)
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

def ag_pca():
    attack,original,labels = get_my_data("attack_base_data/ours/yelp/data.tsv")
    original = original[:500]
    labels = labels[:500]
    embeddings = get_embeddings(attack)
    embeddings_VAT=get_ours_embeddeding(original,labels=labels)
    embeddings_VAT_1=get_VAT_embeddeding(original,labels=labels)
    return embeddings,embeddings_VAT,embeddings_VAT_1

# attack,original,labels = get_my_data("attack_base_data/ours/yelp/data.tsv")
# original = original[:500]
# labels = labels[:500]
#
#
# embeddings = get_embeddings(attack)
# embeddings_VAT=get_ours_embeddeding(original,labels=labels)
# embeddings_VAT_1=get_VAT_embeddeding(original,labels=labels)
#
# labels_1 = np.repeat([0, 1, 2, 3 ,4, 5], repeats=500)
#
# all_embeddings = np.vstack([embeddings, embeddings_VAT_1,embeddings_VAT])  # 假设 embeddings 和 embeddings_VAT 形状一致
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


# attack,original,labels = get_my_data("attack_base_data/ours/yelp/data.tsv")
# original = original[:500]
# labels = labels[:500]
# embeddings = get_embeddings(attack)
# embeddings_VAT=get_ours_embeddeding(original,labels=labels)
# embeddings_VAT_1=get_VAT_embeddeding(original,labels=labels)
#
# labels_1 = np.repeat([0, 1, 2, 3 ,4, 5], repeats=500)
#
# all_embeddings = np.vstack([embeddings, embeddings_VAT_1,embeddings_VAT])  # 假设 embeddings 和 embeddings_VAT 形状一致
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
# class_names = ["TextFooler", "PWWS", "BERT-Attack", "DeepWordBug", "VAT", "Ours"]
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
#
# plt.legend()  # 显示图例
# plt.grid(True)
#
# plt.savefig("embed.png",dpi=500)
#
# plt.show()


