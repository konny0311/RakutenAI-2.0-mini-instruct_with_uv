from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Rakuten/RakutenAI-2.0-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model.eval()

chat = [
    {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
    {"role": "user", "content": "How to make an authentic Spanish Omelette?"},
]

input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device=model.device)
attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
tokens = model.generate(
    input_ids,
    max_length=2048,
    do_sample=False,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=attention_mask,
)
out = tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)
print("ASSISTANT:\n" + out)
print()
