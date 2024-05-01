import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import os
import json

# 加载训练好的模型和 tokenizer
model = MT5ForConditionalGeneration.from_pretrained('./mt5_finetuned')
tokenizer = T5Tokenizer.from_pretrained('./mt5_finetuned')

# 准备评估数据和 DataLoader
def load_data(base_dir, lang, mode):
    data = json.load(open(os.path.join(base_dir, lang, mode + ".json"), encoding="utf-8"))
    return data

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
    
def eval():

    data = load_data()
    inputs = tokenizer([x['input'] for x in data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    dataset = EvalDataset(inputs)
    loader = DataLoader(dataset, batch_size=1)

    # F1 score metric
    f1_metric = load_metric("f1")

    # 评估模型
    model.eval()
    predictions = []
    references = [x['output'] for x in data]

    for batch in loader:
        input_ids = batch['input_ids'].squeeze(1).to(model.device)
        attention_mask = batch['attention_mask'].squeeze(1).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(generated_texts)

    # 计算 F1 分数
    for ref, pred in zip(references, predictions):
        if ref != "O":
            # 更新 F1 metric 实例
            f1_metric.add_batch(predictions=[pred], references=[ref])

    f1_score = f1_metric.compute(average='macro')  # 可以是 'micro', 'macro', 或 'weighted'
    print(f"F1 Score: {f1_score}")