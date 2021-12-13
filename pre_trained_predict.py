import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")

model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-hate-speech")


async def predict(text):
    with torch.no_grad():
        input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        print(input)
        outputs = model(**input)
        print(outputs)
        pred = outputs.logits.argmax(dim=1).item()
        
        return pred
