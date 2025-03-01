from dataloader import BiasDataset
import torch
from collections import defaultdict

class CLRDataset(BiasDataset):
    """
    Dataset to load in contrastive learning.
    Note that the negative and positive samples are pre-built and not transformed as the dataset is called.
    This is because of compute limitations; if we were to augment the anchor sample during train time,
    the forward and backward of the model would be well complete already.

    Thus, just load in whatever skyler gives me lol
    """
    def __getitem__(self, idx):
        '''
        Custom getitem function
        '''
        return_dict = defaultdict(dict)
        item = self.data[idx]

        # labels = {'bias_score': item['score']}  # Dictionary of multi-dimensional bias attributes
        
        for key in ('original_text', 'biased_text', 'paraphrased'):
            text = item[key]
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            for k, v in tokens.items():
                return_dict[k][key] = v
        
        # Convert labels dict to tensor
        return return_dict


