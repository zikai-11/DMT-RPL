
import random

import torch
import torch.nn as nn


from textattack.models.wrappers import ModelWrapper
from transformers import AutoTokenizer

from mask_ids import mask_ids


import modeling_textattack

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# simplify_dict = {'v2': simplifier.simplify_v2,
#                  'random_freq_v1': simplifier.random_freq_v1,
#                  'random_freq_v2': simplifier.random_freq_v2}

model_classfication = modeling_textattack.BertForSequenceClassificationAdvV2.from_pretrained("experiments/MR-base",
                                                                            num_labels=2).cuda()
Mask_ids = mask_ids(threshold=200, syn_num=20, syn_threshold=0.7, ratio=0.3, most_freq_num=10)
simplify1 = Mask_ids.simplify_v2
# simplify2 = Mask_ids.random_freq_v2
class NLI_infer_BERT(ModelWrapper):
    def __init__(self,model,
                 max_seq_length=128,
                 tokenizer=None,num_classes=None):
        super(NLI_infer_BERT, self).__init__()
        self.model = model
        # self.perturb_ratio = perturb_ratio
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        # self.mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
        self.num_classes = num_classes
    def transform_text(self, data, simplify):
        # transform data into seq of embeddings
        text = []
        for (ex_index, text_a) in enumerate(data):
            text_a = simplify(text_a)
            text.append(text_a)


        #
        # # Run prediction for full data
        # eval_sampler = SequentialSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler)

        return text
    # def get_emb_grad(self, input_ids, attention_mask,token_type_ids):
    #     """Get gradient of loss with respect to input tokens.
    #     Args:
    #         text_input (str): input string
    #     Returns:
    #         Dict of ids, tokens, and gradient as numpy array.
    #     """
    #
    #     self.model.eval()
    #     embedding_layer = self.model.bert.embeddings
    #     original_state = embedding_layer.word_embeddings.weight.requires_grad
    #
    #     embedding_layer.word_embeddings.weight.requires_grad = True
    #
    #     emb_grads = []
    #     def grad_hook(module, grad_in, grad_out):
    #         emb_grads.append(grad_out[0])
    #
    #     emb_hook = embedding_layer.word_embeddings.register_full_backward_hook(grad_hook)
    #
    #     self.model.zero_grad()
    #     logits2 = self.model.forward_infer(input_ids, attention_mask,token_type_ids,inference=True)[1]
    #
    #     adv_ids = torch.argmax(logits2,dim=-1)
    #     logits1 = self.model.forward_infer(adv_ids, attention_mask, token_type_ids, inference=True)[0]
    #     pred = torch.argmax(logits1, dim=-1)
    #     loss_fn = nn.CrossEntropyLoss()
    #     loss2 = loss_fn(logits2.view(-1,30522),input_ids.view(-1))
    #     loss1 = loss_fn(logits1.view(-1, self.num_classes), pred.view(-1))
    #
    #     loss = loss1+loss2
    #     loss.backward()
    #
    #     embedding_layer.word_embeddings.weight.requires_grad = original_state
    #     emb_hook.remove() # Remove Hook
    #     self.model.eval()
    #
    #     return emb_grads
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


    def get_masked(self, input_ids, mlm_probability=0.15, label=None):
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
    # def get_new_embedding(self, emb_grads, mask_ids_index, mask_ids, input_ids_expand, length):
    #     # input_ids_expand_embedding = self.model.bert.embeddings.word_embeddings(input_ids_expand)
    #     # mask_ids_embedding = self.model.bert.embeddings.word_embeddings(mask_ids)
    #     delta_grad = emb_grads[0].detach()
    #     norm_grad = torch.norm(delta_grad, p=2, dim=-1)
    #     for i, indices in enumerate(mask_ids_index):
    #         indices = [i for i in indices if i != 0 and i < length]
    #         if indices == []:
    #             continue
    #         importance = norm_grad[:, indices]
    #         min_score = importance.min()
    #         max_score = importance.max()
    #         score = (importance - min_score) / (max_score - min_score)
    #         score = torch.where(score > 0.5, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
    #         one_maxtrix = torch.ones_like(score)
    #
    #         mask_ids[i,indices] = (one_maxtrix-score)[:,:]*input_ids_expand[i,indices]+score[:,:]*mask_ids[i,indices]
    #
    #     new_embedding = mask_ids
    #     return new_embedding
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


    def forward_inference_whole_word_mask1(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 4
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")
        seq_length = len(words)
        mask_size = max(1,int(seq_length * 0.5))
        sample_index = [i for i in range(seq_length)]
        mask_token = self.tokenizer.mask_token
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        mask_index_all = []
        for _ in range(total_expand):
            mask_index = random.sample(sample_index, mask_size)
            tmp_words = words.copy()
            # mask_index1 = mask_index
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
            input_ids_expand = torch.tensor(input_ids_list, dtype=torch.long).to(device='cuda')
            mask_expand = torch.tensor(attention_mask_list, dtype=torch.long).to(device='cuda')
            seg_expand = torch.tensor(token_type_ids_list, dtype=torch.long).to(device='cuda')



            rebuild_ids = self.model.forward_infer(input_ids_expand,attention_mask=mask_expand, token_type_ids=seg_expand, inference=True)[1]
            rebuild_ids1 = torch.argmax(rebuild_ids, dim=-1)
            # print("rebuild_ids11111",rebuild_ids1)

            logits = self.model.forward_infer(rebuild_ids1, mask_expand, seg_expand, inference=True)[0]  # N, num-labels



            logits = l2norm(logits)
            logits = torch.mean(logits, dim=0, keepdim=True)

            return (logits,)

    def forward_inference_mass(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 4
        with torch.no_grad():
            input_ids_expand = input_ids.repeat(total_expand, 1)  # N, L
            mask_expand = attention_mask.repeat(total_expand, 1)  # N, L
            seg_expand = token_type_ids.repeat(total_expand, 1)  # N, L
            length = attention_mask.sum()
            probability_matrix = torch.full(input_ids_expand.shape, 0.2, device=input_ids.device)
            masked_indices = torch.bernoulli(probability_matrix).long()
            masked_indices = masked_indices * 103  # make mask
            masked_indices_nt = masked_indices.eq(0)
            masked_ids = input_ids_expand * masked_indices_nt + masked_indices
            mask_ids_index = torch.nonzero(masked_ids == 103, as_tuple=False)

        # 将索引按照样本进行分组
            mask_ids_by_sample = []
            for i in range(input_ids_expand.size(0)):
                sample_indices = mask_ids_index[mask_ids_index[:, 0] == i][:, 1]
                mask_ids_by_sample.append(sample_indices.tolist())

            # logits_mask_ids = self.model.forward_infer(masked_ids, mask_expand, seg_expand,inference = True)[0]
            with torch.enable_grad():
                embedd_grad = self.get_emb_grad(input_ids, attention_mask, token_type_ids)

            new_embedding = self.get_new_embedding(embedd_grad, mask_ids_by_sample, masked_ids, input_ids_expand,
                                                   length)

            rebuild_ids = self.model.forward_infer(new_embedding, attention_mask=mask_expand, token_type_ids=seg_expand,
                                                   inference=True)[1]

            # rebuild_ids= self.model.forward_infer(input_ids_expand, mask_expand, seg_expand,inference = True)[1]
            rebuild_ids1 = torch.argmax(rebuild_ids, dim=-1)
            # print("rebuild_ids11111",rebuild_ids1)
            logits = self.model.forward_infer(rebuild_ids1, mask_expand, seg_expand, inference=True)[0]  # N, num-labels

            logits = l2norm(logits)
            logits = torch.mean(logits, dim=0, keepdim=True)

            return (logits,)

    def forward_inference_whole_word_mask2(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        total_expand = 4
        input_ids_ = input_ids.repeat(total_expand, 1)
        length = attention_mask.sum()
        text_list = input_ids.cpu().numpy().tolist()[-1]
        sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        words = sentence.split(" ")
        seq_length = len(words)
        mask_size = max(1,int(seq_length * 0.5))
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

            rebuild_ids = self.model.forward_infer(new_embedding,attention_mask=mask_expand, token_type_ids=seg_expand,
                                                   inference=True)[1]

            # rebuild_ids= self.model.forward_infer(input_ids_expand, mask_expand, seg_expand,inference = True)[1]
            rebuild_ids1 = torch.argmax(rebuild_ids, dim=-1)
            # print("rebuild_ids11111",rebuild_ids1)
            logits = self.model.forward_infer(rebuild_ids1, mask_expand, seg_expand, inference=True)[0]  # N, num-labels

            logits = l2norm(logits)
            logits = torch.mean(logits, dim=0, keepdim=True)

            return (logits,)


    def __call__(self,text):
        # Switch the model to eval mode.

        dataloader1 = self.transform_text(text,simplify=simplify1)
        # dataloader2 = self.transform_text(text, simplify=simplify2)
        probs_all = []
        inputs_dict = self.tokenizer(
            dataloader1,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs_dict["input_ids"].clone().detach().to(device='cuda')
        input_mask = inputs_dict["attention_mask"].clone().detach().to(device='cuda')
        segment_ids = inputs_dict["token_type_ids"].clone().detach().to(device='cuda')
        # input_ids = torch.tensor(inputs_dict["input_ids"]).to(device='cuda')
        # input_mask = torch.tensor(inputs_dict["attention_mask"]).to(device='cuda')
        # segment_ids = torch.tensor(inputs_dict["token_type_ids"]).to(device='cuda')

        with torch.no_grad():

                # mask_logits = self.forward_inference_whole_word_mask(input_ids[ii].unsqueeze(0),
                #                                                      input_mask[ii].unsqueeze(0),
                #                                                      segment_ids[ii].unsqueeze(0),
                #                                                      max_seq_len=self.max_seq_length,
                #                                                      perturb_ratio=self.perturb_ratio) # 随机掩码
                # print(mask_logits)
            probs = []
            for (ii,_) in enumerate(input_ids):

                logits_0 = self.forward_inference_whole_word_mask1(input_ids[ii].unsqueeze(0),
                                                   input_mask[ii].unsqueeze(0),
                                                   segment_ids[ii].unsqueeze(0)
                                                   )[0]

                # logits_1 = self.forward_inference_whole_word_mask2(input_ids[ii].unsqueeze(0),
                #                                        input_mask[ii].unsqueeze(0),
                #                                        segment_ids[ii].unsqueeze(0)
                #                                        )[0]
                # logits_1 = self.forward_inference_mass(input_ids[ii].unsqueeze(0),
                #                                         input_mask[ii].unsqueeze(0),
                #                                         segment_ids[ii].unsqueeze(0)
                #                                         )[0]
                # logits_0 = self.forward_inference_whole_word_mask1_mnli(input_ids[ii].unsqueeze(0),
                #                                        input_mask[ii].unsqueeze(0),
                #                                        segment_ids[ii].unsqueeze(0),dataloader=dataloader1[ii]
                #                                        )[0]
                # logits_1 = self.forward_inference_whole_word_mask2_mnli(input_ids[ii].unsqueeze(0),
                #                                        input_mask[ii].unsqueeze(0),
                #                                        segment_ids[ii].unsqueeze(0),dataloader=dataloader1[ii]
                #                                        )[0]
                ############

                mask_logits = logits_0


                # mask_logits = self.model.forward_infer(input_ids[ii].unsqueeze(0),
                #                         input_mask[ii].unsqueeze(0),
                #                         segment_ids[ii].unsqueeze(0),inference =True)[0]
                probs1_mask = nn.functional.softmax(mask_logits, dim=-1)

                probs.append(probs1_mask.squeeze().tolist())
            probs = torch.Tensor(probs).cuda()
            probs_all.append(probs)
            # print(probs_all)

        return (torch.cat(probs_all, dim=0))

