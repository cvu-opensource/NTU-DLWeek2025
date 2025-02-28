from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from transformers import Trainer, TrainingArguments
from transformers import get_scheduler
from torch.optim import AdamW   
import torch
from tqdm.auto import tqdm
from llm2vec.models import LlamaBiModel

from dataloader import BiasDataset, custom_collate_fn


def pretrain_model(data_dir, model_name='Llama-encoder-1.0B', output_dir='./finetune/finetune_results', num_train_epochs=3, batch_size=2):
    '''
    Pretraining.
    '''
    model = LlamaBiModel.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print('model loaded')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Defining dataset split and dataloaders
    dataset = BiasDataset(data_dir, tokenizer, max_length=512)
    # TODO: idk what to do here lol, underneath is the original code
    # train_data, eval_data = random_split(dataset.data, [len(dataset.data) - eval_len, eval_len], generator=torch.Generator())
    # train_data, _ = random_split(dataset.data, [len(dataset.data) - 2, 2], generator=torch.Generator())

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()

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
    pretrain_model(data_dir='./finetune\dataset\clean_with_scores.json')
