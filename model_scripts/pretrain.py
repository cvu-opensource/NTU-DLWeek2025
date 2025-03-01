import sys

sys.path.insert(0, '.')
from clrdata import CLRDataset
from models.clr import CLR
# from models.clr import clr_collate
from llm2vec.models import LlamaBiModel
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch

from dataloader import BiasDataset, TransformModule


def clr_collate(batch):
    flattened_dict = {}
    print(batch)
    for outer_key in batch[0]:
    # For each outer key, iterate over the inner keys (e.g., 'c', 'd')
        print('outerkey', outer_key, batch[0][outer_key].shape)
        for inner_key in batch[0][outer_key]:
            # Create a new combined key (e.g., 'a_c', 'a_d')
            combined_key = f"{outer_key}_{inner_key}"
            print(combined_key)
            print([d[outer_key][inner_key].shape for d in batch])
            # Stack the tensors corresponding to this combined key across all dictionaries
            flattened_dict[combined_key] = torch.stack([d[outer_key][inner_key] for d in batch], dim=0)
            print(flattened_dict[combined_key].shape)

    return flattened_dict

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
    # transforms = TransformModule(
    #     model_name='gpt2',
    #     max_length=2048,
    #     bias_threshold=0.5,
    #     negative_samples=2,
    #     positive_samples=1,
    # )
    dataset = CLRDataset(data_dir, tokenizer, max_length=512)
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
        data_collator=clr_collate,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Pretraining complete. Model saved at", output_dir)


if __name__=='__main__':
    # before you run this pull the requisite model 
    pretrain_model(data_dir='model_scripts/processed_dataset/clean_with_scores.json', batch_size=5)
