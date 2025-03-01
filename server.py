import torch
import json
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.classification import ClassificationWrapper

app = FastAPI()
# RUN USING WINDOWS!! I HATE WSL !!!
# Load the fine-tuned model and tokenizer -> TODO: Change paths as necessary as long as 'config.json' exists
MODEL_NAME = 'Llama-encoder-1.0B'
CHECKPOINT_PATH = r'E:\NTU-DLWeek2025\finetune_results\checkpoint-91'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure ID is correctly set

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.config.pad_token_id = model.config.eos_token_id

# Apply trained adapter modules to basemodel
with open(fr"{CHECKPOINT_PATH}\adapter_config.json", "r") as f:
    adapter_config = json.load(f)
peft_model = PeftModel.from_pretrained(model, f"{CHECKPOINT_PATH}", adapter_config=adapter_config)

device = "cuda" if torch.cuda.is_available() else "cpu"
peft_model.to(device)
serving_model = ClassificationWrapper(peft_model)

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
        outputs = serving_model.infer(**encoding).item()

    return outputs


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
