%%time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompt = """
<|im_start|>system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|>user
### Câu hỏi:
Số gồm 1 đơn vị và 3 chục đọc là :
### Các lựa chọn:
A. 30
B. 31
C. 20
D. 21
### Câu trả lời:

<|im_start|>assistant
""".strip()

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))