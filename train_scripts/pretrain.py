from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW   
import torch
from tqdm.auto import tqdm
from llm2vec.models import LlamaBiModel

from dataloader import BiasDataset, custom_collate_fn


def pretrain_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./train_scripts/pretrain_results', num_train_epochs=3, batch_size=1):
    '''
    Pretraining.
    '''
    # Initialising tokeniser and model 
    model = LlamaBiModel.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print('model loaded')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Defining dataset split and dataloaders
    dataset = BiasDataset(data_dir, tokenizer, max_length=512)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

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
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Pretraining complete. Model saved at", output_dir)


if __name__=='__main__':
    pretrain_model(data_dir='./train_scripts\dataset\clean_with_scores.json')
