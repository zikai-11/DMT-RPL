
import random

import torch
import torch.nn as nn


from textattack.models.wrappers import ModelWrapper
from transformers import AutoTokenizer

from mask_ids import mask_ids

import numpy as np


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X




Mask_ids = mask_ids(threshold=200, syn_num=20, syn_threshold=0.7, ratio=0.3, most_freq_num=10)
simplify1 = Mask_ids.simplify_v2
# simplify2 = Mask_ids.random_freq_v2
class NLI_infer_BERT(ModelWrapper):
    def __init__(self,model,
                 max_seq_length=128,
                 tokenizer=None,num_classes=None,model1=None):
        super(NLI_infer_BERT, self).__init__()
        self.model = model
        # self.perturb_ratio = perturb_ratio
        self.max_seq_length = max_seq_length
        self.model1 = model1
        self.tokenizer = tokenizer
        # self.mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
        self.num_classes = num_classes
    def transform_text(self, data, simplify):
        # transform data into seq of embeddings
        text = []
        for (ex_index, text_a) in enumerate(data):
            text_a = simplify(text_a)
            text.append(text_a)
        return text

    def get_emb_grad(self, input_ids, attention_mask,token_type_ids):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        self.model.eval()
        embedding_layer = self.model.bert.embeddings
        original_state = embedding_layer.word_embeddings.weight.requires_grad

        embedding_layer.word_embeddings.weight.requires_grad = True

        emb_grads = []
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.word_embeddings.register_full_backward_hook(grad_hook)
        #register_full_backward_hook
        self.model.zero_grad()
        # mask_ids,_= self.get_masked(input_ids)
        # logits_mlm = self.model.forward_infer(mask_ids, attention_mask,token_type_ids,inference=True)[1]
        #
        # mlm_ids = torch.argmax(logits_mlm,dim=-1)
        logits_cls = self.model.forward_infer(input_ids, attention_mask,token_type_ids,inference=True)[0]
        pred = torch.argmax(logits_cls, dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss1 = loss_fn(logits_cls.view(-1, self.num_classes), pred.view(-1))
        loss = loss1
        loss.backward()

        embedding_layer.word_embeddings.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook
        self.model.eval()

        return emb_grads


    def get_new_embedding(self, emb_grads, mask_ids_index, mask_ids, input_ids_expand, length):
        delta_grad = emb_grads[0].detach()
        norm_grad = torch.norm(delta_grad, p=2, dim=-1)

        for i, indices in enumerate(mask_ids_index):
            indices = [idx for idx in indices if idx != 0 and idx < length]
            if not indices:
                continue
            importance = norm_grad[:, indices]
            # Soft thresholding and adaptive weighting
            score = torch.sigmoid(importance - importance.mean(dim=-1, keepdim=True))
            score = torch.where(score > 0.5, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
            # Adaptive weighting
            weighted_input = (1 - score) * input_ids_expand[i, indices]
            weighted_mask = score * mask_ids[i, indices]

            # Update mask_ids
            mask_ids[i, indices] = weighted_input + weighted_mask

        new_embedding = mask_ids
        return new_embedding


    def forward_inference_whole_word_mask1(self, input_ids, attention_mask=None, token_type_ids=None, input_ids_original=None,
                                           attention_mask_original=None,token_type_ids_original=None):
        total_expand = 4
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")

        text_list_original = input_ids_original.cpu().numpy().tolist()[-1]
        sentence_original = self.tokenizer.decode(text_list_original, skip_special_tokens=True)
        words_original = sentence_original.split(" ")

        different_indices = [i for i, (w1, w2) in enumerate(zip(words, words_original)) if w1 != w2]

        per_word = [words[idx] for idx in different_indices]

        seq_length = len(words)
        mask_size = int(seq_length * 0.5)
        sample_index = [i for i in range(seq_length)]
        mask_token = self.tokenizer.mask_token
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        mask_index_all = []
        for _ in range(total_expand):
            mask_index = random.sample(sample_index, mask_size)
            # mask_index_all.append(mask_index)
            tmp_words = words.copy()
            mask_index1 = Mask_ids.mask_frequency(tmp_words,mask_index)
            for index in mask_index1:
                sub_word = self.tokenizer(tmp_words[index], add_special_tokens=False)["input_ids"]
                mask_length = len(sub_word)
                mask_words = " ".join([mask_token for _ in range(mask_length)])
                tmp_words[index] = mask_words
            tokenizer_result = self.tokenizer.encode_plus(" ".join(tmp_words), None,
                                                          add_special_tokens=True, padding="max_length",
                                                          max_length=self.max_seq_length, truncation=True)
            input_ids_list.append(tokenizer_result["input_ids"])
            attention_mask_list.append(tokenizer_result["attention_mask"])
            token_type_ids_list.append(tokenizer_result["token_type_ids"])

        with torch.no_grad():
            input_ids_expand = torch.tensor(input_ids_list, dtype=torch.long).to(device='cuda')
            mask_expand = torch.tensor(attention_mask_list, dtype=torch.long).to(device='cuda')
            seg_expand = torch.tensor(token_type_ids_list, dtype=torch.long).to(device='cuda')
        sub_list = []
        for ids in input_ids_expand:
            sentence_res = self.tokenizer.decode(ids, skip_special_tokens=True)
            sentence_res_words = sentence_res.split(" ")
            mask_number = len(words)-len(sentence_res_words)

            intersection_res = set(per_word) & set(sentence_res_words) #取交集

            if mask_number != 0:
                sub = (len(per_word) - len(intersection_res)) / mask_number
                sub_list.append(sub)
            else:
                sub = 0

        per = np.mean(sub_list)
        return per,sub_list

    def forward_inference_whole_word_mask2(self, input_ids, attention_mask=None, token_type_ids=None, input_ids_original=None,
                                           attention_mask_original=None,token_type_ids_original=None):
        total_expand = 4

        input_ids_ = input_ids.repeat(total_expand, 1)
        length = attention_mask.sum()
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")

        text_list_original = input_ids_original.cpu().numpy().tolist()[-1]
        sentence_original = self.tokenizer.decode(text_list_original, skip_special_tokens=True)
        words_original = sentence_original.split(" ")

        different_indices = [i for i, (w1, w2) in enumerate(zip(words, words_original)) if w1 != w2]

        seq_length = len(words)
        mask_size = int(seq_length * 0.5)
        sample_index = [i for i in range(seq_length)]
        mask_token = self.tokenizer.mask_token
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        mask_index_all = []
        for _ in range(total_expand):
            mask_index = random.sample(sample_index, mask_size)
            # mask_index_all.append(mask_index)
            tmp_words = words.copy()
            for index in mask_index:
                sub_word = self.tokenizer(tmp_words[index], add_special_tokens=False)["input_ids"]
                mask_length = len(sub_word)
                mask_words = " ".join([mask_token for _ in range(mask_length)])
                tmp_words[index] = mask_words
            tokenizer_result = self.tokenizer.encode_plus(" ".join(tmp_words), None,
                                                          add_special_tokens=True, padding="max_length",
                                                           max_length=self.max_seq_length, truncation=True)
            input_ids_list.append(tokenizer_result["input_ids"])
            attention_mask_list.append(tokenizer_result["attention_mask"])
            token_type_ids_list.append(tokenizer_result["token_type_ids"])

        with torch.no_grad():
            input_ids_expand = torch.tensor(input_ids_list, dtype=torch.long).to(device='cuda')
            mask_expand = torch.tensor(attention_mask_list, dtype=torch.long).to(device='cuda')
            seg_expand = torch.tensor(token_type_ids_list, dtype=torch.long).to(device='cuda')

            # ###加嵌入
            mask_ids_index = torch.nonzero(input_ids_expand == 103, as_tuple=False)

            # 将索引按照样本进行分组
            mask_ids_by_sample = []
            for i in range(input_ids_expand.size(0)):
                sample_indices = mask_ids_index[mask_ids_index[:, 0] == i][:, 1]
                mask_ids_by_sample.append(sample_indices.tolist())

            # logits_mask_ids = self.model.forward_infer(masked_ids, mask_expand, seg_expand,inference = True)[0]
            with torch.enable_grad():
                embedd_grad = self.get_emb_grad(input_ids, attention_mask, token_type_ids)

            new_embedding = self.get_new_embedding(embedd_grad, mask_ids_by_sample, input_ids_expand, input_ids_,
                                                   length)


            sub_list = []
            per_word = [words[idx] for idx in different_indices]



            for ids in new_embedding:

                sentence_res = self.tokenizer.decode(ids, skip_special_tokens=True)
                sentence_res_words = sentence_res.split(" ")
                mask_number = len(words) - len(sentence_res_words)

                intersection_res = set(per_word) & set(sentence_res_words)
                if mask_number != 0:
                    sub = (len(per_word) - len(intersection_res)) / mask_number
                    sub_list.append(sub)
                else:
                    sub = 0

            per = np.mean(sub_list)

        return per
    def forward_inference_mass(self, input_ids, attention_mask=None, token_type_ids=None, input_ids_original=None,
                               attention_mask_original=None, token_type_ids_original=None):
        total_expand = 8
        a = 0
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")

        text_list_original = input_ids_original.cpu().numpy().tolist()[-1]
        sentence_original = self.tokenizer.decode(text_list_original, skip_special_tokens=True)
        words_original = sentence_original.split(" ")

        different_indices = [i for i, (w1, w2) in enumerate(zip(words, words_original)) if w1 != w2]

        per_word = [words[idx] for idx in different_indices]

        # 比较两个张量，生成布尔掩码
        # mask_not_equal = torch.ne(input_ids, input_ids_original)
        #
        # # 找到不同的索引
        # indices_different = torch.nonzero(mask_not_equal, as_tuple=False)

        with torch.no_grad():
            input_ids_expand = input_ids.repeat(total_expand, 1)  # N, L
            mask_expand = attention_mask.repeat(total_expand, 1)  # N, L
            seg_expand = token_type_ids.repeat(total_expand, 1)  # N, L

            probability_matrix = torch.full(input_ids_expand.shape, 0.2, device=input_ids.device)
            masked_indices = torch.bernoulli(probability_matrix).long()
            masked_indices = masked_indices * 103  # make mask
            masked_indices_nt = masked_indices.eq(0)
            masked_ids = input_ids_expand * masked_indices_nt + masked_indices
            sub_list = []
            per_word_ids = self.tokenizer(" ".join(per_word))["input_ids"]
            for ids in masked_ids:
                ids1 = ids.unsqueeze(0)

                # 获取 attention_mask 中的有效长度
                max_count = 128
                ids1_list = ids1.cpu().numpy().tolist()[-1]
                # 初始化计数器
                count_103 = 0

                # 遍历 ids1 和 attention_mask，只计算前 max_count 个位置
                for i in range(max_count):
                    if attention_mask[0, i]:  # 如果 attention_mask 是 True，则考虑该位置
                        if ids1[0, i] == 103:
                            count_103 += 1
                mask_number = count_103
                # sentence_res = self.tokenizer.decode(ids, skip_special_tokens=True)
                # sentence_res_words = sentence_res.split(" ")
                # mask_number = len(words) - len(sentence_res_words)

                intersection_res = set(per_word_ids) & set(ids1_list)

                if mask_number != 0:
                    sub = (len(per_word_ids) - len(intersection_res)) / mask_number
                    sub_list.append(sub)
                else:
                    sub = 0
            per = np.mean(sub_list)

        return per




    def __call__(self,text_original,text_attack):

        inputs_dict1 = self.tokenizer(
            text_original,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_origial = inputs_dict1["input_ids"].clone().detach().to(device='cuda')
        input_mask_origial = inputs_dict1["attention_mask"].clone().detach().to(device='cuda')
        segment_ids_origial = inputs_dict1["token_type_ids"].clone().detach().to(device='cuda')

        inputs_dict2 = self.tokenizer(
            text_attack,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_attack = inputs_dict2["input_ids"].clone().detach().to(device='cuda')
        input_mask_attack = inputs_dict2["attention_mask"].clone().detach().to(device='cuda')
        segment_ids_attack = inputs_dict2["token_type_ids"].clone().detach().to(device='cuda')


        with torch.no_grad():
                freq = []
                important = []
                mass = []
                for (ii, _) in enumerate(input_ids_origial):
                    logits_0,sub_list = self.forward_inference_whole_word_mask1(input_ids_attack[ii].unsqueeze(0),
                                                                       input_mask_attack[ii].unsqueeze(0),
                                                                       segment_ids_attack[ii].unsqueeze(0),
                                                                       input_ids_origial[ii].unsqueeze(0),
                                                                       input_mask_origial[ii].unsqueeze(0),
                                                                       segment_ids_origial[ii].unsqueeze(0),

                                                                       )

                    logits_1 = self.forward_inference_whole_word_mask2( input_ids_attack[ii].unsqueeze(0),
                                                                       input_mask_attack[ii].unsqueeze(0),
                                                                       segment_ids_attack[ii].unsqueeze(0),
                                                                        input_ids_origial[ii].unsqueeze(0),
                                                                       input_mask_origial[ii].unsqueeze(0),
                                                                       segment_ids_origial[ii].unsqueeze(0),
                                                                       )

                    logits_2 = self.forward_inference_mass( input_ids_attack[ii].unsqueeze(0),
                                                                       input_mask_attack[ii].unsqueeze(0),
                                                                       segment_ids_attack[ii].unsqueeze(0),
                                                                       input_ids_origial[ii].unsqueeze(0),
                                                                       input_mask_origial[ii].unsqueeze(0),
                                                                       segment_ids_origial[ii].unsqueeze(0), )


                    if sub_list!=[]:
                        freq.append(logits_0)

                    important.append(logits_1)
                    mass.append(logits_2)
                percent_freq = np.mean(freq)
                percent_important = np.mean(important)
                percent_mass  = np.mean(mass)

        return percent_freq,percent_important,percent_mass

