import torch
import json
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Cache
from transformers import Trainer
from transformers import TrainingArguments
from peft import get_peft_model
from peft import LoraConfig
from peft import TaskType
from models.classification import ClassificationWrapper
from dataloader import BiasDataset, custom_collate_fn


def finetune_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./model_scripts/finetune_results', num_train_epochs=50, batch_size=4, split_ratio=0.9):
    '''
    Fine-tuning pretrained model with PEFT
    '''
    # Initialising model, classifier and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 1 for bias and nonbias classes
    model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)
    model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.classifier.bias.data.zero_()

    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = True  # Ensure caching is enabled
    model.config.cache_class = Cache  # Explicitly use the new cache class

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure ID is correctly set

    # Defining dataset split and dataloaders
    dataset = BiasDataset(data_dir, tokenizer, max_length=512)
    
    eval_len = int(max(1, (1 - split_ratio) * len(dataset)))
    train_data, eval_data = random_split(dataset, [len(dataset.data) - eval_len, eval_len], generator=torch.Generator())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
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

    # Defining training configs
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=len(dataset),
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=None,
        label_names=["labels"],  # Explicitly define label names
        remove_unused_columns=False,
    )
    
    model.config.use_cache = False
    model = ClassificationWrapper(model)

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
    model.peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete. Model saved at", output_dir)

    # testing whether model actually outputs correctly
    tests = ['for the sake of testing', 'i think he is a very very bad person', 'he was a well respected person']
    for test in tests:
        encoding = tokenizer(
            test,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            scores = model.infer(**encoding).item()
            print(test, scores)


if __name__=='__main__':
    model = finetune_model(data_dir='./dataset', output_dir='./finetune_results', num_train_epochs=2, batch_size=2)
