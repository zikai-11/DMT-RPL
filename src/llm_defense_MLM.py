
import random

import torch
import torch.nn as nn
import transformers

from textattack.models.wrappers import ModelWrapper
from transformers import AutoTokenizer

from mask_ids import mask_ids
import modeling_textattack


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

Mask_ids = mask_ids(threshold=200, syn_num=20, syn_threshold=0.7, ratio=0.3, most_freq_num=10)
simplify1 = Mask_ids.simplify_v2

class NLI_infer_BERT():

    def __init__(self,model=None,
                 max_seq_length=None,
                 tokenizer=None,num_classes=None):
        super(NLI_infer_BERT, self).__init__()
        self.device = "cuda:0"
        self.model = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/imdb-adv",
                                                                                            num_labels=2).to(device=self.device)

        self.max_seq_length = 256
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        self.num_classes = 2

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

        norm_grad = emb_grads

        for i, indices in enumerate(mask_ids_index):
            indices = [idx for idx in indices if idx != 0 and idx < length-1]
            if not indices:
                continue

            importance = norm_grad[:, indices]

            # Soft thresholding and adaptive weighting
            score = torch.sigmoid(importance - importance.mean(dim=-1, keepdim=True))
            score = torch.where(score > 0.5, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
            # Adaptive weighting
            weighted_input = (1 - score) * input_ids_expand[i, indices]
            weighted_mask = score * mask_ids[i, indices]

            # Update mask_ids
            mask_ids[i, indices] = weighted_input + weighted_mask

        new_embedding = mask_ids
        return new_embedding

    def forward_inference_whole_word_mask1(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 4
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")
        seq_length = len(words)
        mask_size = max(1,int(seq_length * 0.3))
        sample_index = [i for i in range(seq_length)]
        mask_token = self.tokenizer.mask_token
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        for _ in range(total_expand):
            mask_index = random.sample(sample_index, mask_size)
            tmp_words = words.copy()
            #mask_index1 = mask_index
            mask_index1 = Mask_ids.mask_frequency(tmp_words,mask_index)
            for index in mask_index1:
                sub_word = self.tokenizer(tmp_words[index], add_special_tokens=False)["input_ids"]
                mask_length = len(sub_word)
                mask_words = " ".join([mask_token for _ in range(mask_length)])
                tmp_words[index] = mask_words
            tokenizer_result = self.tokenizer.encode_plus(" ".join(tmp_words), None,
                                    add_special_tokens=True,padding="max_length",max_length=self.max_seq_length, truncation=True)
            input_ids_list.append(tokenizer_result["input_ids"])
            attention_mask_list.append(tokenizer_result["attention_mask"])
            token_type_ids_list.append(tokenizer_result["token_type_ids"])

        with torch.no_grad():
            input_ids_expand = torch.tensor(input_ids_list, dtype=torch.long).to(device=self.device)
            mask_expand = torch.tensor(attention_mask_list, dtype=torch.long).to(device=self.device)
            seg_expand = torch.tensor(token_type_ids_list, dtype=torch.long).to(device=self.device)



            rebuild_ids = self.model.forward_infer(input_ids_expand,attention_mask=mask_expand, token_type_ids=seg_expand, inference=True)[1]
            rebuild_ids1 = torch.argmax(rebuild_ids, dim=-1)

            # 将 rebuild_ids1 从 tensor 转换为 numpy 数组
            rebuild_ids1 = rebuild_ids1.cpu().numpy()
            model_inputs = []
            for sentence_ids in rebuild_ids1:
                model_input = self.tokenizer.decode(sentence_ids, skip_special_tokens=True)
                model_inputs.append(model_input)

            return model_inputs

    def forward_inference_whole_word_mask2(self, input_ids, attention_mask=None,sentence=None,token_type_ids=None, labels=None,word_grad=None,mlm_tokenizer=None):
        total_expand = 4
        input_ids_ = input_ids.repeat(total_expand, 1)
        length = attention_mask.sum()
        # text_list = input_ids.cpu().numpy().tolist()[-1]
        # sentence = mlm_tokenizer.decode(text_list, skip_special_tokens=True)

        # words = sentence.split(" ")
        # seq_length = len(words)
        # mask_size = max(1,int(seq_length * 0.3))
        # sample_index = [i for i in range(seq_length)]


        # 添加自定义 MASK token
        # 将 [MASK] 添加为普通 token
        mlm_tokenizer.add_tokens(["[MASK]"])
        # 可选：显式设置 mask_token 属性
        mlm_tokenizer.mask_token = "[MASK]"
        probability_matrix = torch.full(input_ids_.shape, 0.3, device=input_ids.device)
        masked_indices = torch.bernoulli(probability_matrix).long()
        masked_indices = masked_indices * 32000  # make mask
        masked_indices_nt = masked_indices.eq(0)
        input_ids_expand = input_ids_ * masked_indices_nt + masked_indices

        mask_token = mlm_tokenizer.mask_token
        mask_token_ids = mlm_tokenizer.convert_tokens_to_ids(mask_token)
        # input_ids_list = []
        # attention_mask_list = []
        # token_type_ids_list = []
        # for _ in range(total_expand):
        #     mask_index = random.sample(sample_index, mask_size)
        #     # mask_index_all.append(mask_index)
        #     tmp_words = words.copy()
        #     for index in mask_index:
        #         sub_word = mlm_tokenizer(tmp_words[index], add_special_tokens=False)["input_ids"]
        #         if sub_word == []:
        #             continue
        #         mask_length = len(sub_word)
        #         mask_words = " ".join([mask_token for _ in range(mask_length)])
        #         tmp_words[index] = mask_words
        #     tokenizer_result = mlm_tokenizer(" ".join(tmp_words),
        #                                      add_special_tokens=True,
        #                                      return_tensors="pt", padding=True, truncation=True)
        #
        #     input_ids_list.append(tokenizer_result["input_ids"])
        #     attention_mask_list.append(tokenizer_result["attention_mask"])
        #
        #
        with torch.no_grad():
        #     input_ids_expand = torch.cat(input_ids_list).to(device=self.device)
        #     mask_expand = torch.cat(attention_mask_list).to(device=self.device)


            # ###加嵌入
            mask_ids_index = torch.nonzero(input_ids_expand == int(mask_token_ids), as_tuple=False)

            # 将索引按照样本进行分组
            mask_ids_by_sample = []
            for i in range(input_ids_expand.size(0)):
                sample_indices = mask_ids_index[mask_ids_index[:, 0] == i][:, 1]
                mask_ids_by_sample.append(sample_indices.tolist())

            # logits_mask_ids = self.model.forward_infer(masked_ids, mask_expand, seg_expand,inference = True)[0]

            embedd_grad = word_grad

            new_ids = self.get_new_embedding(embedd_grad, mask_ids_by_sample, input_ids_expand, input_ids_,
                                                   length)
            #special_tokens = mlm_tokenizer.special_tokens_map
            special_tokens_to_remove = [token for token in mlm_tokenizer.special_tokens_map.values() if token!="[MASK]"]
            #special_tokens_to_remove = mlm_tokenizer.special_tokens

            mlm_inputs = []
            for ids in new_ids:
                mlm_input = mlm_tokenizer.decode(ids, skip_special_tokens=False)
                for token in special_tokens_to_remove:
                    mlm_input = mlm_input.replace(token, '')
                mlm_input = ' '.join(mlm_input.split())

                mlm_inputs.append(mlm_input)


            encoded_inputs = self.tokenizer(mlm_inputs,add_special_tokens=True,
                                                return_tensors = "pt",padding="max_length",max_length=self.max_seq_length, truncation = True).to(device=self.device)

            input_ids_MLM = encoded_inputs["input_ids"]

            attention_mask_MLM = encoded_inputs["attention_mask"]


            rebuild_ids = self.model.forward_infer(input_ids_MLM,attention_mask=attention_mask_MLM,
                                                   inference=True)[1]

            # rebuild_ids= self.model.forward_infer(input_ids_expand, mask_expand, seg_expand,inference = True)[1]
            rebuild_ids1 = torch.argmax(rebuild_ids, dim=-1)
            # print("rebuild_ids11111",rebuild_ids1)
            # 将 rebuild_ids1 从 tensor 转换为 numpy 数组
            rebuild_ids1 = rebuild_ids1.cpu().numpy()
            model_inputs = []
            for sentence_ids in rebuild_ids1:
                model_input = self.tokenizer.decode(sentence_ids, skip_special_tokens=True)
                model_inputs.append(model_input)

            return model_inputs



    def defense(self,text,word_grad,mlm_tokenizer):

        probs_all = []
        inputs_dict = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",

        )
        input_ids = inputs_dict["input_ids"].clone().detach().to(device=self.device)
        input_mask = inputs_dict["attention_mask"].clone().detach().to(device=self.device)
        segment_ids = inputs_dict["token_type_ids"].clone().detach().to(device=self.device)

        inputs_dict_mlm = mlm_tokenizer(
            text,
            add_special_tokens=True,
            return_tensors = "pt", padding = True, truncation = True
        )


        input_ids_mlm = inputs_dict_mlm["input_ids"].clone().detach().to(device=self.device)
        input_mask_mlm = inputs_dict_mlm["attention_mask"].clone().detach().to(device=self.device)
        # segment_ids_mlm = inputs_dict_mlm["token_type_ids"].clone().detach().to(device='cuda')

        with torch.no_grad():

            probs = []
            for (ii,_) in enumerate(input_ids):

                logits_0 = self.forward_inference_whole_word_mask1(input_ids[ii].unsqueeze(0),
                                                   input_mask[ii].unsqueeze(0),
                                                   segment_ids[ii].unsqueeze(0)
                                                   )

                logits_1 = self.forward_inference_whole_word_mask2(input_ids_mlm[ii].unsqueeze(0),
                                                       input_mask_mlm[ii].unsqueeze(0)
                                                       ,word_grad=word_grad,mlm_tokenizer=mlm_tokenizer,sentence=text)

                ############

                logits = logits_0+logits_1

        return logits

