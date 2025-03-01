from dataloader import BiasDataset
import torch
import random
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split

def dino_collate_fn(batch):
    """
    Woohoo

    Squash along the batch dimension to produce on unified dictionary with one input_ids, attention_mask 
    keys, and values are N times size long now.

    Args:
        batch: list of dicts.
        [
            {
                'global_crops': [
                    global_crop_tokens_1, global_crop_tokens_2, ...
                ],  # but N times the size as before see

                'local_crops': [
                    local_crop_tokens_1, local_crop_tokens_2, ...
                ],  # but N times the size as before see
            },
            {
                'global_crops': [
                    global_crop_tokens_1, global_crop_tokens_2, ...
                ],  # but N times the size as before see

                'local_crops': [
                    local_crop_tokens_1, local_crop_tokens_2, ...
                ],  # but N times the size as before see
            },
            ... N times
        ]


    Returns:
    dictionary of lists:
        {
            'global_crops': [
                global_crop_tokens_1, global_crop_tokens_2, ...
            ],  # but N times the size as before see

            'local_crops': [
                local_crop_tokens_1, local_crop_tokens_2, ...
            ],  # but N times the size as before see
        }
    """
    return_dict = defaultdict(list)
    for dict in batch:
        for key, value in dict.items():
            for crop in value:
                return_dict[key].append(crop)
                print('return dict key', key, 'crop shape', crop.shape)

    return return_dict

class DINOTransformModule:
    def __init__(
            self, 
            tokenizer,
            max_length=512,
            global_crops=2,
            local_crops=8,
            global_crop_ratio=[0.14, 1],
            local_crop_ratio=[0.05, 0.14],
        ):
        """
        Transforms to do random cropping of the article.
        """
        self.max_length = max_length
        self.global_crops = global_crops
        self.local_crops = local_crops
        self.global_crop_ratio = global_crop_ratio
        self.local_crop_ratio = local_crop_ratio

    def __call__(self, text):
        """
        Main function to call for applying transformations.

        Transforms for either local/global crops are as follows:
        - Sample from a range that is provided at initialization
        - Do the crop
        - Tokenize
        - Format it however i like

        Return:
            dictionary of lists:
            {
                'global_crops': [
                    global_crop_1, global_crop_2, ...
                ],

                'local_crops': [
                    local_crop_1, local_crop_2, ...
                ],
            }
        """
        return_dict = {
            'global_crops': [],
            'local_crops': [],
        }
        for i in range(self.global_crops):
            ratio = random.uniform(self.global_crop_ratio[0], self.global_crop_ratio[1])
            percentage = int(ratio * 100)
            return_dict['global_crops'].append(self.random_crop_article(text, percentage))

        for i in range(self.local_crops):
            ratio = random.uniform(self.local_crop_ratio[0], self.local_crop_ratio[1])
            percentage = int(ratio * 100)
            return_dict['local_crops'].append(self.random_crop_article(text, percentage))
        
        return return_dict
    
    @staticmethod
    def random_crop_article(text, percentage):
        """
        Crops the text at a random position based on the given percentage.
        Args:
            text: The full article as a string.
            percentage: The percentage of the article to crop (0-100).
        Return:
            A cropped portion of the article.
        """
        # Validate the percentage
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100.")

        # Calculate the total length of the crop
        crop_length = int(len(text) * (percentage / 100))
        print(crop_length)
        print(len(text))

        # Ensure crop length is greater than 0
        if crop_length == 0:
            raise ValueError("Crop length must be greater than 0. Try increasing the percentage.")

        # Get a random start index
        max_start_index = len(text) - crop_length
        if max_start_index <= 0:
            return text  # Return the entire text if it's too small to crop
        
        start_index = random.randint(0, max_start_index)

        # Crop the text from the random start index
        cropped_text = text[start_index:start_index + crop_length]

        return cropped_text


class DinoDataset(BiasDataset):
    """
    Dataset to do dino things.
    Dino demands we crop the input into global and local crops; that will be handled by transform module.
    
    Return:
            dictionary of lists:
            {
                'global_crops': [
                    global_crop_tokens_1, global_crop_tokens_2, ...
                ],

                'local_crops': [
                    local_crop_tokens_1, local_crop_tokens_2, ...
                ],
            }
    """
    def __getitem__(self, idx):
        '''
        Custom getitem function
        '''
        return_dict = defaultdict(list)
        item = self.data[idx]
        text = item['text']

        if len(text) > 0:

            global_local_dict = self.transforms(text)
            
            for k, ls in global_local_dict.items():
                for text in ls:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return_dict[k].append(encoding)
        
        return return_dict

data_tokenizer = AutoTokenizer.from_pretrained('/mnt/e/NTU-DLWeek2025/gpt2')
data_tokenizer.pad_token_id = data_tokenizer.eos_token_id

# Defining dataset split and dataloaders
transforms = DINOTransformModule(
    tokenizer=data_tokenizer,
    max_length=512,
    global_crops=2,
    local_crops=8,
    global_crop_ratio=[0.14, 1],
    local_crop_ratio=[0.05, 0.14],
)
dataset = DinoDataset('/mnt/e/NTU-DLWeek2025/model_scripts/datasets/clean_with_scores.json', data_tokenizer, max_length=512, transforms=transforms)
train_dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=dino_collate_fn)

for batch in train_dataloader:
    pass
    # print("batch", batch)