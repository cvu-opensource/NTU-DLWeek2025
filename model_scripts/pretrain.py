import sys

sys.path.insert(0, '.')

from models.clr import CLR
from llm2vec.models import LlamaBiModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW   
import torch
from tqdm.auto import tqdm

from dataloader import BiasDataset, custom_collate_fn, TransformModule


def pretrain_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./model_scripts/pretrain_results', num_train_epochs=3, batch_size=1):
    '''
    Pretraining.
    '''
    # Initialising tokeniser and model 
    print('loading model')
    model = LlamaBiModel.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print('loaded model')
    clr = CLR(llama_model=model, infonce_reduction='mean')
    print('clr built')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Defining dataset split and dataloaders
    transforms = TransformModule(
        model_name='gpt2',
        max_length=2048,
        bias_threshold=0.5,
        negative_samples=2,
        positive_samples=1,
    )
    dataset = BiasDataset(data_dir, tokenizer, max_length=512, transforms=transforms)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

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
        remove_unused_columns=False,
    )
    
    torch.cuda.empty_cache()

    # Defining trainer
    trainer = Trainer(
        model=clr,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Pretraining complete. Model saved at", output_dir)


if __name__=='__main__':
    # before you run this pull the requisite model 
    pretrain_model(data_dir='model_scripts/datasets/clean_with_scores.json', batch_size=5)
