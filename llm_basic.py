from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain what machine learning is in one sentence."}
    ],
    temperature=0.2,
    max_tokens=50
)

print(response.choices[0].message.content)
