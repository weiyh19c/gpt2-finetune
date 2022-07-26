import random

def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index


!pip install pytorch_transformers==1.0  # 安装 PyTorch-Transformers

import torch
from pytorch_transformers import GPT2Tokenizer

import logging
logging.basicConfig(level=logging.INFO)

# 载入预训练模型的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 使用 GPT2Tokenizer 对输入进行编码
text = "Yesterday, a man named Jack said he saw an alien,"
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])
tokens_tensor.shape


from pytorch_transformers import GPT2LMHeadModel

# 读取 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained("./")
model.eval()

total_predicted_text = text
n = 100  # 预测过程的循环次数
for _ in range(n):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)

    if '<|endoftext|>' in total_predicted_text:
        # 如果出现文本结束标志，就结束文本生成
        break

    indexed_tokens += [predicted_index]
    tokens_tensor = torch.tensor([indexed_tokens])

print(total_predicted_text)


with open('./romeo_and_juliet.txt', 'r') as f:
    dataset = f.read()

len(dataset)


indexed_text = tokenizer.encode(dataset)
del(dataset)

dataset_cut = []
for i in range(len(indexed_text)//512):
    # 将字符串分段成长度为 512
    dataset_cut.append(indexed_text[i*512:i*512+512])
del(indexed_text)

dataset_tensor = torch.tensor(dataset_cut)
dataset_tensor.shape


from torch.utils.data import DataLoader, TensorDataset

# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=2,
                          shuffle=False)
train_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 30  # 循环学习 30 次

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(
            target).to(device)

        optimizer.zero_grad()

        loss, logits, _ = model(data, labels=target)

        total_loss += loss

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('average loss:', total_loss/len(train_loader))

print('训练时间：', time.time()-pre)


text = "From fairest creatures we desire"  # 这里也可以输入不同的英文文本
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])

model.eval()
total_predicted_text = text

# 使训练后的模型进行 500 次预测
for _ in range(500):
    tokens_tensor = tokens_tensor.to('cuda')

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = select_top_k(predictions, k=10)

    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)
    if '<|endoftext|>' in total_predicted_text:
        # 如果出现文本结束标志，就结束文本生成
        break

    indexed_tokens += [predicted_index]

    if len(indexed_tokens) > 1023:
        # 模型最长输入长度为1024，如果长度过长则截断
        indexed_tokens = indexed_tokens[-1023:]

    tokens_tensor = torch.tensor([indexed_tokens])

print(total_predicted_text)
