# import csv
# import os
# texts = []
# labels = []
# with open("data/sst/original_data/test.tsv", "r", encoding="utf-8-sig") as f:
#     content = csv.reader(f,delimiter='\t')
#     for row in content:
#         if row[0] == 'sentence':
#             continue
#         texts.append(row[0])
#         labels.append(row[1])
# print(texts[0],labels[0])
# with open(os.path.join("data/sst/original_data/" ,'test'), 'w') as ofile:
#     for texts,labels in zip(texts,labels):
#             ofile.write('{} {}\n'.format(labels,texts))
import torch

# a= [0,1]
# tensor1 = torch.Tensor([0,1])
# tensor2 = torch.Tensor([[[1,2,3],[2,3,4],[1,2,3]],[[1,2,2],[3,4,3],[1,2,3]]])
# tensor3 = torch.Tensor([[[1,1,1],[2,2,2],[2,2,2]],[[3,3,3],[6,6,5],[3,3,3]]])
# print(tensor2.shape)
# one = torch.ones_like(tensor1)
# tensor2[0,a,:]=tensor1.unsqueeze(0)[:,:,None]*tensor2[0,a,:]+(one-tensor1).unsqueeze(0)[:,:,None]*tensor3[0,a,:]
# tensor2[1,a,:]=tensor1.unsqueeze(0)[:,:,None]*tensor2[1,a,:]+(one-tensor1).unsqueeze(0)[:,:,None]*tensor3[1,a,:]
#
# print(tensor2)
# tensor3 = torch.transpose(tensor2.squeeze(),0,1)
# ten = torch.matmul(tensor2.squeeze(),tensor3)
# print(ten)
# print(ten.diagonal())
x = torch.tensor([2.0]).squeeze()
y = torch.quantile(x,0.5)
print(x.shape)