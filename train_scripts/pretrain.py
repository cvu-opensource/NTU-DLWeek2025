from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.optim import AdamW   
import torch
from tqdm.auto import tqdm
# from datasets import load_metric

from dataloader import BiasDataset, custom_collate_fn


def pretrain_model(data_dir, model_name='Llama-encoder-1.0B'):
    '''
    Pretraining.
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Defining dataset split and dataloaders
    dataset = BiasDataset(data_dir, tokenizer, max_length=512)
    
    eval_len = int(max(1, (1 - split_ratio) * len(dataset)))
    train_data, eval_data = random_split(dataset.data, [len(dataset.data) - eval_len, eval_len], generator=torch.Generator())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    # tokenizer.pad_token = tokenizer.eos_token
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding="max_length", truncation=True)

    # tokenized_train = small_train_dataset.map(tokenize_function, batched=True)

    # tokenized_train = tokenized_train.remove_columns(["text"])
    # tokenized_train = tokenized_train.rename_column("label", "labels")

    # train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=1)


    # Load model and train configs
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=5)
    model.config.pad_token_id = model.config.eos_token_id
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    # model.model.gradient_checkpointing = True

    # Iterating epochs
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            preprocess_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    preprocess_batch[k] = v.to(device)
                elif isinstance(v, list):
                    preprocess_batch[k] = torch.stack(v, dim=1).to(device)

            for k, v in preprocess_batch.items():
                print(k,v.shape)
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**preprocess_batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


if __name__=='__main__':
    pretrain_model(data_dir='./finetune\dataset\clean_with_scores.json')
