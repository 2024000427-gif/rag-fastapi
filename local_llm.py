from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

prompt = "I am a "

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,              # enable randomness
    temperature=0.8,             # allow variation
    repetition_penalty=1.3,      # penalize repeated phrases
    top_k=50,                    # limit choices
    top_p=0.95                   # nucleus sampling
)


result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
