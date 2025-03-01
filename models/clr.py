from torch import nn
import torch
from llm2vec.models import LlamaBiModel
from info_nce import InfoNCE, info_nce

class CLR(nn.Module):
    """
    Wrapper around LlamaBiModel (the llmtovec package) for contrastive learning of representation SSL training.
    For sheer convenience, we use the infoNCE package that someone already wrote for. We roughly know the theory,
    just that implementation is a hassle for such a short time.
    """

    def __init__(self, llama_model: LlamaBiModel, infonce_reduction='mean'):
        #TODO document or not lol
        super().__init__()
        llama_model.config.pad_token_id = llama_model.config.eos_token_id
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        llama_model.to(device)
        self.infoNCEloss = InfoNCE(reduction=infonce_reduction)
        self.llama_model = llama_model


    def forward(self, batch: dict):
        """
        Args:
            - batch: Dictionary from dataloader of structure:  
                {
                    'input_ids': dict of 2-dim tensors e.g {
                        'original_text': (N,D), 
                        'paraphrased': (N,D), 
                        'biased_text': (M,D)
                    },
                        
                    'attention_mask': same shape as input_ids.
                }
        """

        anchor = self.llama_model(
            input_ids=batch['input_ids']['anchor'],
            attention_mask=batch['input_ids']['anchor'],
        )

        positive = self.llama_model(
            input_ids=batch['input_ids']['positive'],
            attention_mask=batch['input_ids']['positive'],
        )
        
        negative = self.llama_model(
            input_ids=batch['input_ids']['negative'],
            attention_mask=batch['input_ids']['negative'],
        )

        # compute loss here instead of seperate function, so don't pass to hf trainer
        loss = self.infoNCEloss(
            query=anchor,
            positive_key=positive,
            negative_keys=negative
        )

        print('nce loss', loss)
        return loss  # backwards done for us by trainer