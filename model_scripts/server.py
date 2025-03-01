import torch
import json
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classification import ClassificationWrapper

app = FastAPI()

MODEL_NAME = 'Llama-encoder-1.0B'
CHECKPOINT_PATH = './saved_results/checkpoint-91'

# Load the fine-tuned model and tokenizer -> TODO: Change paths as necessary as long as 'config.json' exists
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure ID is correctly set

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = ClassificationWrapper(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class TextInput(BaseModel):
    text: str


def get_bias_score(text: str) -> float:
    """
    Function that calls our model to get bias score for a given text
    """
    # Tokenize the input text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(device)

    # Get the model's prediction (bias score)
    with torch.no_grad():
        scores = model.infer(**encoding).item()
    return scores


@app.post("/predict_bias/")
async def predict_bias(input_data: TextInput):
    """
    Endpoint to predict the bias score for an input text
    """
    try:
        bias_score = get_bias_score(input_data.text)
        return {"bias_score": bias_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    """
    Health check endpoint to verify the server is up and running
    """
    return {"message": "API is running."}


# Sample implementation
if __name__=='__main__':
    tests = ['for the sake of testing', 'i think he is a very very bad person', 'he was a well respected person']
    for test in tests:
        print(get_bias_score(test))
