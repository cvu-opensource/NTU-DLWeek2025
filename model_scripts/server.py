from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Load the fine-tuned model and tokenizer -> TODO: Change paths as necessary
MODEL_PATH = './finetune_results'
MODEL_NAME = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

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
        outputs = model(**encoding)
        bias_score = outputs.logits.item()  # Assuming the output logits are directly the bias score
    return bias_score


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
