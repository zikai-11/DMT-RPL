import random

from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.modeling_bert import BertForSequenceClassification, SequenceClassifierOutput, BertModel, \
    BertPreTrainedModel, BertOnlyMLMHead
from transformers import RobertaModel,RobertaForSequenceClassification
from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaClassificationHead
from transformers import ElectraModel,ElectraForSequenceClassification
from transformers.modeling_electra import ElectraPreTrainedModel,ElectraClassificationHead
from transformers import PretrainedConfig, AutoConfig
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import numpy as np
from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaClassificationHead,RobertaLMHead,RobertaForMaskedLM


# from transformers import AutoModel, AutoTokenizer
# import torch
# import torch.nn as nn
# from transformers import AutoModelForSequenceClassification, BertModel
# from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertPreTrainedModel, \
#     BertOnlyMLMHead
# from transformers import RobertaModel, RobertaForSequenceClassification
# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaClassificationHead
# from transformers import ElectraModel, ElectraForSequenceClassification
# from transformers.models.electra.modeling_electra import ElectraPreTrainedModel, ElectraClassificationHead
# from torch.nn import CrossEntropyLoss, MSELoss

def real_labels(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    isMR_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] > 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 10
        elif labels[i] > 7:
            orig_labels[i] = labels[i] - 8
        elif labels[i] > 5:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] > 3:
            simplify_labels[i] = 1
            isMR_labels[i] = 1
            attack_labels[i] = 0
            orig_labels[i] = labels[i] - 4
        elif labels[i] > 1:
            attack_labels[i] = 1
            isMR_labels[i] = 1
            orig_labels[i] = labels[i] - 2
        elif labels[i] > -1:
            isMR_labels[i] = 0
            orig_labels[i] = labels[i]
    return attack_labels,orig_labels, simplify_labels, isMR_labels

def real_labels_agnews(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    isMR_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        isMR_labels[i] = 0
        orig_labels[i] = labels[i]
    return attack_labels,orig_labels, simplify_labels, isMR_labels

def real_labels_mnli(labels):
    attack_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    orig_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    simplify_labels = torch.zeros_like(labels, dtype=labels.dtype, device=labels.device)
    for i in range(len(labels)):
        if labels[i] < 3:
            orig_labels[i] = labels[i]
        elif labels[i] < 6 :
            orig_labels[i] = labels[i] - 3
            simplify_labels[i] = 1
        elif labels[i] < 9:
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 6
        elif labels[i] < 12:
            simplify_labels[i] = 1
            attack_labels[i] = 1
            orig_labels[i] = labels[i] - 9
    return attack_labels, orig_labels, simplify_labels



from transformers.activations import get_activation



class RoBertaForSequenceClassificationAdv(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super(RoBertaForSequenceClassificationAdv, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.mlm_probability = 0.15
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier1 = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.adv_steps = 2
        self.adv_init_mag = 1e-1  # 2e-1
        self.adv_max_norm = 2e-1  # 5e-1
        self.adv_lr = 1e-2  ## 5e-2
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        mask_ids, output_labels = self.get_masked(input_ids)
        adv_ids,tr_loss = self.adversarial_training(input_ids=input_ids,
                                                attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels,
                                                mask_ids=mask_ids,output_labels=output_labels)

        mask_ids, output_labels = self.get_masked_adv(adv_ids, input_ids_adv=input_ids)
        outputs = self.roberta(mask_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        mask_output = self.lm_head(sequence_output)
        logits1 = self.classifier1(sequence_output)  # 输出为2 分类器
        loss = 0
        if labels is not None:
            loss = self.compute_loss(logits1=logits1,mask_output=mask_output,output_labels=output_labels,labels=labels)
        all_loss = tr_loss+loss
        loss.backward()

        return all_loss

    def initialize_delt(self,embeds_init):
        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)  # 随机初始化一个和
        dims = torch.tensor(768, device=delta.device).float()
        mag = self.adv_init_mag / torch.sqrt(dims)  # B

        delta = delta * mag.view(-1, 1, 1)
        return delta

    def adversarial_training(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            labels,
            mask_ids,
            output_labels

    ):
        embeds_init = self.roberta.embeddings.word_embeddings(mask_ids)

        delta = self._initialize_delta(embeds_init)
        tr_loss = 0
        for astep in range(self.adv_steps):

            delta.requires_grad_()
            inputs_embeds = embeds_init + delta

            outputs = self.roberta(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            # print(input_ids.shape)
            sequence_output = outputs[0]


            mask_output = self.lm_head(sequence_output)
            logits1 = self.classifier1(sequence_output)
            loss = 0
            if labels is not None:
                loss = self.compute_loss(logits1=logits1,mask_output=mask_output,output_labels=output_labels,labels=labels)

            if astep == self.adv_steps - 1:
                break
            tr_loss += loss
            loss.backward()

            # get grad on delta
            delta_grad = delta.grad.clone().detach()

            delta = self.update_delta(input_ids,delta_grad,embeds_init)

        adv_ids = self.get_adv_ids(logits=mask_output)

        return adv_ids,tr_loss


    def update_delta(self,input_ids,delta_grad,embeds_init):

        bs, seq_len = input_ids.size()
        # grad-norm
        denorm = torch.norm(delta_grad, dim=-1).view(bs, seq_len, 1)  # B seq-len 1
        denorm = torch.clamp(denorm, min=1e-8)
        # add the delta with grads
        delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # B seq-len D

        # normalize new delta at token-level
        delta_norm = torch.norm(delta, p=2, dim=-1).detach()  # B seq-len
        mean_norm, _ = torch.max(delta_norm, dim=-1, keepdim=True)  # B,1
        # reweight-delta using scaling
        reweights_tok = (delta_norm / mean_norm).view(bs, seq_len, 1)  # B seq-len, 1
        delta = delta * reweights_tok

        # reweight the exceed delta
        delta_norm = torch.norm(delta.view(bs, -1).float(), p=2, dim=1).detach()
        exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
        reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)  # B 1 1

        # detach delta and embeds init for next iteration
        delta = (delta * reweights).detach()

        return delta

    def compute_loss(self,logits1,mask_output,output_labels,labels):
        attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels_agnews(labels)
        # active_loss3 = mask_frequency_labels.view(-1) == 0
        loss_fct1 = CrossEntropyLoss()
        # print(attack_labels)
        loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))
        loss_fct3 = CrossEntropyLoss()

        loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))
        loss = loss1 + loss3

        return loss

    def get_masked(self, input_ids, mlm_probability=0.15, label=None):
        output_labels = input_ids.clone()
        input_ids = input_ids.clone()
        # output_labels[output_labels == pad_token_id] = -100
        probability_matrix = torch.full(input_ids.shape, mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        output_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(output_labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 50264  # hard code mask index

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(output_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # hard code random word
        random_words = torch.randint(self.config.vocab_size, output_labels.shape, dtype=torch.long, device=output_labels.device)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, output_labels

    def get_adv_ids(self, logits):
        values, indices = torch.max(logits, dim=-1)
        return indices

    def get_masked_adv(self, input_ids, mlm_probability=0.15, input_ids_adv=None, label=None):
        output_labels = input_ids_adv.clone()
        input_ids = input_ids.clone()
        # output_labels[output_labels == pad_token_id] = -100
        probability_matrix = torch.full(input_ids.shape, mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        output_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(output_labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 50264  # hard code mask index

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(output_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # hard code random word
        random_words = torch.randint(self.config.vocab_size, output_labels.shape, dtype=torch.long, device=output_labels.device)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, output_labels

    def forward_infer(self,
                      input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      labels=None,
                      output_attentions=None,
                      output_hidden_states=None,
                      return_dict=None,
                      inference=False, ):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask, token_type_ids=token_type_ids)

        output_labels = input_ids.clone()
        sequence_output = outputs[0]

        mask_output = self.lm_head(sequence_output)  # 返回的是vocab的大小[1,128,30522]

        logits1 = self.classifier1(sequence_output)  # 输出为2 分类器
        # logits2 = self.classifier2(pooled_output)  # 输出为1 检测器
        # print(logits1)
        # prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels_agnews(labels)
            # active_loss3 = mask_frequency_labels.view(-1) == 0

            loss_fct1 = CrossEntropyLoss()
            # print(attack_labels)
            loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))
            loss_fct3 = CrossEntropyLoss()
            loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))

            loss = loss1 + loss3

        if inference:
            output = (logits1, mask_output) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (loss, logits1)


class BertForSequenceClassificationAdvV2(BertForMaskedLM):
    def __init__(self, config):
        super(BertForSequenceClassificationAdvV2,self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.mlm_probability = 0.15
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """


        tr_loss = 0
        adv_steps = 5
        adv_init_mag = 5e-2 #1e-1
        adv_max_norm = 1    #2e-1
        adv_lr = 3e-2       ## 1e-2

        # print("input",input_ids)
        mask_ids,output_labels  = self.get_masked(input_ids)
        embeds_init = self.bert.embeddings.word_embeddings(mask_ids)

        delta = torch.zeros_like(embeds_init).uniform_(-1, 1)  # 随机初始化一个和
        dims = torch.tensor(768, device=delta.device).float()
        mag = adv_init_mag / torch.sqrt(dims)  # B
        bs, seq_len = input_ids.size()
        delta = delta * mag.view(-1, 1, 1)

        for astep in range(adv_steps):
            delta.requires_grad_()
            inputs_embeds = embeds_init + delta

            outputs = self.bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            # print(input_ids.shape)
            sequence_output = outputs[0]
            pooled_output = outputs[1]

            mask_output = self.cls(sequence_output)
            pooled_output = self.dropout(pooled_output)
            logits1 = self.classifier1(pooled_output)  # 输出为2 分类器
            loss = 0
            if labels is not None:
                attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels_agnews(labels)
                #active_loss3 = mask_frequency_labels.view(-1) == 0
                loss_fct1 = CrossEntropyLoss()
                #print(attack_labels)
                loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))
                loss_fct3 = CrossEntropyLoss(ignore_index=-100)

                loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))
                loss = loss3+loss1

            tr_loss += loss

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
            embeds_init = self.bert.embeddings.word_embeddings(mask_ids)

        return mask_output


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

    def get_adv_ids(self, logits):
        # print(logits)

        values, indices = torch.max(logits,dim=-1)

        # indices 包含最大值的索引，形状为 [16, 128, 1]

        # 您可能需要压缩最后一个维度，因为 topk 返回的索引会包含一个维度为 k 的维度
        # indices = indices.squeeze(-1)
        return indices

    def forward_adv(self, mask_out,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    labels=None):

        adv_ids = self.get_adv_ids(logits=mask_out) #离散的 梯度不会传输 但是后面的cls会更新
        # print(adv_ids)
        mask_ids, output_labels = self.get_masked_adv(adv_ids, input_ids_adv=input_ids)
        outputs = self.bert(mask_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        mask_output = self.cls(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # 输出为2 分类器
        loss = 0
        if labels is not None:
            attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels_agnews(labels)
            # active_loss3 = mask_frequency_labels.view(-1) == 0
            loss_fct1 = CrossEntropyLoss()
            # print(attack_labels)
            loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))
            loss_fct3 = CrossEntropyLoss(ignore_index=-100)

            loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))

            loss = loss1 + loss3

        loss.backward()

        return loss

    def get_masked_adv(self, input_ids, mlm_probability=0.15,input_ids_adv=None, label=None):
        output_labels = input_ids_adv.clone()
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

    def forward_infer(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            inference=False,):

        mask_ids = input_ids
        outputs = self.bert(mask_ids,
                            attention_mask=attention_mask,token_type_ids=token_type_ids)

        output_labels = input_ids.clone()
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        mask_output = self.cls(sequence_output)  # 返回的是vocab的大小[1,128,30522]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # 输出为2 分类器
        #logits2 = self.classifier2(pooled_output)  # 输出为1 检测器

        #prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels_agnews(labels)
            # active_loss3 = mask_frequency_labels.view(-1) == 0

            loss_fct1 = CrossEntropyLoss()
            # print(attack_labels)
            loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))
            loss_fct3 = CrossEntropyLoss(ignore_index=-100)
            loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))

            loss = loss1+loss3

        if inference:
            output = (logits1,mask_output) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (loss,logits1)

class BertForSequenceClassificationAdvV2_mnli(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 3)
        self.classifier2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inference=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, simplify_labels= real_labels_mnli(labels)
            loss_fct1 = CrossEntropyLoss()
            loss1 = loss_fct1(logits1.view(-1, 3), orig_labels.view(-1))
            loss_fct2 = nn.BCEWithLogitsLoss()
            active_loss2 = simplify_labels.view(-1) == 0
            active_logits2 = logits2.view(-1)[active_loss2]
            active_labels2 = attack_labels.float().view(-1)[active_loss2]
            loss2 = loss_fct2(active_logits2, active_labels2)
            loss = loss1+loss2

        if inference:
            output = (logits1, prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=outputs[0],
            attentions=outputs.attentions,
        )


