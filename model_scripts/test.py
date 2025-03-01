import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForCausalLM
from pathlib import Path
from tqdm import tqdm

# Define input and output directories
INPUT_DIR = Path("./dataset")
OUTPUT_DIR = Path("./processed_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8

# Load model and tokenizer (change model_name to use different models)
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure ID is correctly set
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate text using model
def generate_texts(prompts, max_length=2096):

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=max_length,
            do_sample=True,   # Enables faster stochastic sampling
            top_p=0.9,        # Nucleus sampling to speed up output
            temperature=0.7    # Adds slight randomness for variation
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Process JSON files
for json_file in INPUT_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f'Processing {json_file}'):
        batch = data[i:i+BATCH_SIZE]

        # Modify text fields
        for entry in batch:
            entry["original_text"] = entry.pop("text")

        # Generate biased texts
        bias_prompts = [f"Generated a biased overview of the piece of text, giving a biased interpretation on its ideas and main points. Here is the article: {entry['original_text']}" for entry in batch]
        biased_outputs = generate_texts(bias_prompts)

        # Generate paraphrased texts
        paraphrase_prompts = [f"Generate a paraphrase of the original_text but do not alter the original intentions and ideas of the article. Here is the article: {entry['original_text']}" for entry in batch]
        paraphrased_outputs = generate_texts(paraphrase_prompts)

        # Save results
        for entry, biased_text, paraphrased_text in zip(batch, biased_outputs, paraphrased_outputs):
            entry["biased_text"] = biased_text.strip('Generated a biased overview of the piece of text, giving a biased interpretation on its ideas and main points. Here is the article:')
            entry["paraphrased"] = paraphrased_text.strip('Generate a paraphrase of the original_text but do not alter the original intentions and ideas of the article. Here is the article:')
            print(1, entry['original_text'])
            print(2, entry['biased_text'])
            print(3, entry['paraphrased'])
    
    # Save modified JSON to output directory
    output_file = OUTPUT_DIR / json_file.name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {json_file.name} and saved to {output_file}")
