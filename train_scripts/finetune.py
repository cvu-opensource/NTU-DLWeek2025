import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType

from dataloader import BiasDataset, custom_collate_fn


def finetune_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./finetune/finetune_results', num_train_epochs=3, batch_size=2, split_ratio=0.9):
    '''
    Fine-tuning function with PEFT
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

    # Defining dataset split and dataloaders
    dataset = BiasDataset(data_dir, tokenizer, max_length=512)
    
    eval_len = int(max(1, (1 - split_ratio) * len(dataset)))
    train_data, eval_data = random_split(dataset.data, [len(dataset.data) - eval_len, eval_len], generator=torch.Generator())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialising tokeniser and model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Apply LoRA using PEFT
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)
    model.to(device)  # MODEL TO GPU/CUDA

    # Defining training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=None,
    )
    
    torch.cuda.empty_cache()

    # Defining trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,  # Pass eval dataset
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete. Model saved at", output_dir)


if __name__=='__main__':
    finetune_model(data_dir='./finetune\dataset\clean_with_scores.json')
