import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
from transformers import AdamW
import os
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载与预处理
def load_data(base_dir, lang, mode, dataset):
    data = json.load(open(os.path.join(base_dir, dataset, lang, mode + ".json"), encoding="utf-8"))
    return data

# 准备数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def evaluate(model, valid_loader, tokenizer):
    model.eval()
    f1_metric = load_metric('seqeval')
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="eval"):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Convert decoded predictions and labels to the format expected by 'seqeval'
            for (pred, label) in zip (decoded_preds, decoded_labels):
                if label != 'O':
                    predictions.append([pred])
                    references.append([label])

    # Calculate F1 score
    results = f1_metric.compute(predictions=predictions, references=references)
    return results['overall_f1']


def train(src_lang, tgt_lang, dataset):

    train_batch_size = 4
    eval_batch_size = 4
    learning_rate = 5e-5
    epoch_num = 1
    best_f1 = 0
    k_steps = 1  # 每 k 步执行一次验证
    step = 0

    # 使用Hugging Face的transformers库加载tokenizer和model
    tokenizer = T5Tokenizer.from_pretrained('./mt5-base')
    model = MT5ForConditionalGeneration.from_pretrained('./mt5-base')
    model.to(device)

    # 数据编码
    train_data = load_data("./data", src_lang, "train", dataset)
    train_inputs = tokenizer([x['input'] for x in train_data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    train_outputs = tokenizer([x['output'] for x in train_data], padding=True, truncation=True, max_length=512, return_tensors="pt")

    valid_data = load_data("./data", tgt_lang, "dev", dataset)
    valid_inputs = tokenizer([x['input'] for x in valid_data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    valid_outputs = tokenizer([x['output'] for x in valid_data], padding=True, truncation=True, max_length=512, return_tensors="pt")


    # 创建数据加载器
    dataset = CustomDataset(train_inputs, train_outputs.input_ids)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    valid_dataset = CustomDataset(valid_inputs, valid_outputs.input_ids)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)


    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练模型
    model.train()
    for epoch in tqdm(range(epoch_num)):
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1).to(model.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(model.device)
            labels = batch['labels'].to(model.device)

            train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = train_outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            if step % k_steps == 0:
                f1_score = evaluate(model, valid_loader, tokenizer)
                print(f"Step {step}, Validation F1 Score: {f1_score}")
                if f1_score > best_f1:
                    best_f1 = f1_score
                    model.save_pretrained("./model/best_mt5_model")
                    tokenizer.save_pretrained("./model/best_mt5_model")
                    print("Saved best model based on F1 score")
                    break
            step += 1


if __name__ == "__main__":
    src_lang = "English"
    tgt_lang = "Chinese"
    dataset = "ACE"
    train(src_lang, tgt_lang, dataset)


    tokenizer = T5Tokenizer.from_pretrained(f"./model/best_mt5_model_{src_lang}_{tgt_lang}")
    model = MT5ForConditionalGeneration.from_pretrained(f"./model/best_mt5_model_{src_lang}_{tgt_lang}")
    pred_data = load_data("./data", tgt_lang, "test")
    pred_inputs = tokenizer([x['input'] for x in pred_data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    pred_outputs = tokenizer([x['output'] for x in pred_data], padding=True, truncation=True, max_length=512, return_tensors="pt")
    pred_dataset = CustomDataset(pred_inputs, pred_outputs.input_ids[:, 0].tolist())
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

    test_f1 = evaluate(model, pred_loader, tokenizer)

    print(test_f1)
