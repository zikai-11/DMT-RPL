import random

import transformers
from transformers.models.auto import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.models.bert.modeling_bert import BertForSequenceClassification, SequenceClassifierOutput, BertModel, \
    BertPreTrainedModel, BertOnlyMLMHead ,BertForMaskedLM
from transformers import RobertaModel,RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaClassificationHead
from transformers import ElectraModel,ElectraForSequenceClassification
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel,ElectraClassificationHead
from transformers import PretrainedConfig, AutoConfig
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
# from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
# import torch
# import torch.nn as nn
# from transformers import AutoModelForSequenceClassification, AutoModel
# from transformers.modeling_bert import BertForSequenceClassification, SequenceClassifierOutput, BertModel, \
#     BertPreTrainedModel, BertOnlyMLMHead
# from transformers import RobertaModel,RobertaForSequenceClassification
# from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaClassificationHead
# from transformers import ElectraModel,ElectraForSequenceClassification
# from transformers.modeling_electra import ElectraPreTrainedModel,ElectraClassificationHead
# from transformers import PretrainedConfig, AutoConfig
# from torch.nn import CrossEntropyLoss, MSELoss

class BertForSequenceClassificationAdvV2(BertForMaskedLM):
    def __init__(self, config):
        super(BertForSequenceClassificationAdvV2,self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.mlm_probability = 0.15
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels)
        #self.classifier2 = nn.Linear(config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.cls = BertOnlyMLMHead(config)
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
        mask_ids, output_labels = self.get_masked(input_ids)  # 可以处理多个数据

        outputs = self.bert(
            mask_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,

        )

        # print(input_ids.shape)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # mask_output = self.mask_output(mask_ids,attention_mask,token_type_ids)  # 返回的是vocab的大小[1,128,30522]
        mask_output = self.cls(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # 输出为2 分类器
        # logits2 = self.classifier2(pooled_output)  # 输出为1 检测器
        # prob = torch.sigmoid(logits2)
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

        return (loss, logits1)

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
        # text_list = input_ids.cpu().numpy().tolist()[-1]
        # sentence = self.tokenizer.decode(text_list, skip_special_tokens=True)
        # print("sentense", sentence)
        outputs = self.bert(mask_ids,
                            attention_mask=attention_mask,token_type_ids=token_type_ids,inputs_embeds=inputs_embeds)


        sequence_output = outputs[0]
        pooled_output = outputs[1]

        mask_output = self.cls(sequence_output)  # 返回的是vocab的大小[1,128,30522]

        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)  # 输出为2 分类器
        #logits2 = self.classifier2(pooled_output)  # 输出为1 检测器

        #prob = torch.sigmoid(logits2)
        loss = None
        if labels is not None:
            attack_labels, orig_labels, mask_labels, mask_frequency_labels = real_labels(labels)
            # active_loss3 = mask_frequency_labels.view(-1) == 0

            loss_fct1 = CrossEntropyLoss()

            # print(attack_labels)
            loss1 = loss_fct1(logits1.view(-1, self.config.num_labels), orig_labels.view(-1))

            loss_fct3 = CrossEntropyLoss()

            loss3 = loss_fct3(mask_output.view(-1, self.config.vocab_size), output_labels.view(-1))

            loss = loss1 + loss3

        if inference:
            output = (logits1,mask_output) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (logits1,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (loss,logits1)

    def get_masked(self, input_ids, mlm_probability=0.15, label=None):
        output_labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        #output_labels[~masked_indices] = -100  # We only compute loss on masked tokens
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