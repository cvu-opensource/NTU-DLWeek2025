import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# Define input and output directories
INPUT_DIR = Path("./dataset")
OUTPUT_DIR = Path("./processed_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model and tokenizer (change model_name to use different models)
model_name = "Qwen-1.5B"  # Example model, replace with "deepseek-ai/deepseek-llm-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate text using model
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Process JSON files
for json_file in INPUT_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for entry in data:
        # Rename "text" to "original_text"
        entry["original_text"] = entry.pop("text")
        
        # Generate biased text
        bias_prompt = ("i will provide you a piece of text. "
                       "Generated a biased overview of the piece of text, giving a biased interpretation on its ideas and main points. "
                       f"Here is the article: {entry['original_text']}")
        entry["biased_text"] = generate_text(bias_prompt)
        
        # Generate paraphrased text
        paraphrase_prompt = ("i will provide you a piece of text. "
                             "Generate a paraphrase of the 'original_text' but do not alter the original intentions and ideas of the article. "
                             f"Here is the article: {entry['original_text']}")
        entry["paraphrased"] = generate_text(paraphrase_prompt)
    
    # Save modified JSON to output directory
    output_file = OUTPUT_DIR / json_file.name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {json_file.name} and saved to {output_file}")
