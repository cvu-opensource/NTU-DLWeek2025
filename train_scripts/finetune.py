import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import pipeline
from transformers import Trainer
from transformers import TrainingArguments
from peft import get_peft_model
from peft import LoraConfig
from peft import TaskType

from dataloader import BiasDataset, custom_collate_fn


def classify_sample(sample, num_sentences=3, bias_threshold=0.5):
    """
    Classifies a sample as biased or not based on extracted features
    """
    sentences = sample.split(". ")
    
    if len(sentences) == 1:
        selected_sentences = sentences
    else:
        selected_sentences = random.sample(sentences, min(num_sentences, len(sentences)))
    
    # Load sentiment analysis pipeline (or use a fine-tuned bias detection model)
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiments = sentiment_analyzer(selected_sentences)
    
    # Compute a bias score based on sentiment intensity
    bias_scores = [abs(s["score"]) if s["label"] == "NEGATIVE" else 0 for s in sentiments]
    avg_bias_score = np.mean(bias_scores)
    
    return {
        "bias_score": avg_bias_score,
        "biased": avg_bias_score > bias_threshold
    }


def finetune_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./finetune/finetune_results', num_train_epochs=3, batch_size=2, split_ratio=0.9):
    '''
    Fine-tuning function with PEFT.
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure ID is correctly set

    data_collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    # Defining dataset split and dataloaders
    analyzer = None  # or None
    dataset = BiasDataset(data_dir, tokenizer, analyzer=analyzer, max_length=512)
    
    eval_len = int(max(1, (1 - split_ratio) * len(dataset)))
    train_data, eval_data = random_split(dataset, [len(dataset.data) - eval_len, eval_len], generator=torch.Generator())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    for batch in train_dataloader:
        print(batch)

    # Initialising tokeniser and model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # Apply LoRA using PEFT
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Adjust based on your model
    )
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = model.config.eos_token_id

    # Defining training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=None,
        label_names=["labels"],  # Explicitly define label names
    )
    
    torch.cuda.empty_cache()

    # Defining trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,  # Pass eval dataset
        # tokenizer=tokenizer,
        data_collator=custom_collate_fn
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete. Model saved at", output_dir)


if __name__=='__main__':
    finetune_model(data_dir='./train_scripts\dataset\clean_with_scores.json')
