from .dataloader import BiasDataset
import torch

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
        item = self.data[idx]
        text = item['text']

        labels = {'bias_score': item['score']}  # Dictionary of multi-dimensional bias attributes
        
        # Tokenize all the texts
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels dict to tensor
        label_tensor = torch.tensor(list(labels.values()), dtype=torch.float32)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # ensure this is 2-dim
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 2-dim also
            'labels': label_tensor  # 1-dim tensor (also just 1 value ig)
        }