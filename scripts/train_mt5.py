import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
from transformers import AdamW
import os
import json
from tqdm import tqdm

# 数据加载与预处理
def load_data(base_dir, lang, mode):
    data = json.load(open(os.path.join(base_dir, lang, mode + ".json"), encoding="utf-8"))
    return data

# 准备数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
    
def train(lang):
    # 使用Hugging Face的transformers库加载tokenizer和model
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

    # 数据编码
    data = load_data("./data", lang, "train")
    inputs = tokenizer([x['input'] for x in data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = tokenizer([x['output'] for x in data], padding=True, truncation=True, max_length=512, return_tensors="pt")

    # 创建数据加载器
    dataset = CustomDataset(inputs, outputs.input_ids[:, 0].tolist())
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 训练模型
    model.train()
    for epoch in tqdm(range(5)):
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1).to(model.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # 保存模型
    model.save_pretrained(f"./model/{lang}/mt5_finetuned")
    tokenizer.save_pretrained(f"./model/{lang}/mt5_finetuned")

if __name__ == "__main__":
    for lang in tqdm(["English", "Chinese", "Arabic"]):
        print("train English.")
        train(lang)
