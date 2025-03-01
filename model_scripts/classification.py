import torch
from torch import nn


class ClassificationWrapper(nn.Module):
    """
    Custom hardcoded wrapping around our own things
    I really just want to not bother with making a seperate class but for neatness wtv
    """
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        """
        fyi the first element is the ground truth
        """
        outputs = self.peft_model(*args, **kwargs)
        logits = outputs.logits
        scores = self.softmax(logits)
        labels = kwargs['labels']
        # print('scores', scores)
        # print('labels', labels)
        loss = self.loss(scores, labels)
        # print('loss', loss)
        return {
            'loss': loss,
        }
    
    def infer(self, *args, **kwargs):
        """
        Call this when you want to get results. Hardcoded asf.
        oh well!
        """
        outputs = self.peft_model(*args, **kwargs)
        logits = outputs.logits
        scores = self.softmax(logits)
        print('scores', scores)
        return scores[:, 0]  # wow! amazing!
    