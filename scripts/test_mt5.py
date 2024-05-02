from transformers import T5Tokenizer, MT5ForConditionalGeneration

# 加载模型和 tokenizer
model_name = './mt5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# 准备输入
prompt = "Classify event type: Even as the secretary of homeland security was putting his people on high alert last month, a 30-foot Cuban patrol boat with four heavily armed men landed on American shores, utterly undetected by the Coast Guard Secretary Ridge now leads The event type of landed is"

# 对输入进行编码
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

# 生成回复
outputs = model.generate(input_ids, max_length=100, num_beams=5)

# 解码生成的文本
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded_output)