# pip install openai transformers torch

import openai
from openai import OpenAI

# Used my new key. 
client = OpenAI(
    api_key = "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"
)

from transformers import AutoTokenizer
import torch

tokenizers = {
    "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
    "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "DistilBERT": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    "RoBERTa": AutoTokenizer.from_pretrained("roberta-base")
}

text = "Hello World! Our research is about honorifics."

def tokenize_with_transformers():
    for name, tokenizer in tokenizers.items():
        encoded = tokenizer.encode(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded[0])
       
        print(f"\n{name} Tokenizer")
        print("Encoded tokens:", encoded)
        print("Decoded text:", decoded)

# This doesn't work: maybe all gpt-3.5 or gpt-4, ... requires payment T.T
def tokenize_with_openai():
    """Using OpenAI API"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
   
    tokens_used = response['usage']['total_tokens']
    print("\nOpenAI GPT Tokenization")
    print(f"Total tokens used: {tokens_used}")

    """
    print("\nOpenAI GPT-4 Tokenizer")
    print("Response content:", response['choices'][0]['message']['content'])
    """

def return_tokenize_with_transformers(input_text, model='BERT'):
    # List-up tokenizers
    tokenizers = {
        "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
        "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "DistilBERT": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
        "RoBERTa": AutoTokenizer.from_pretrained("roberta-base")
    }

    tokenizer = tokenizers[model]
    encoded = tokenizer.encode(input_text, return_tensors="pt")
    decoded = tokenizer.decode(encoded[0])

    return encoded


if __name__ == "__main__":
    print("Tokenization using Hugging Face Transformers:")
    tokenize_with_transformers()
    
    # print("\nTokenization using OpenAI API:")
    # tokenize_with_openai()
    # print(return_tokenize_with_openai('안녕하세요. NLP 과제중입니다.', model='BERT'))