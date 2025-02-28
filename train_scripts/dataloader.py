import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of data with multi-dimensional labels.
    """
    # Extract input_ids and attention_mask from the batch
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Handle labels (assumes each label is a tensor of the same size)
    labels = torch.stack([item['labels'] for item in batch])
    
    # Return a batch containing padded input_ids, attention_mask, and labels
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class BiasDataset(Dataset):
    def __init__(self, root, tokenizer, max_length=512):
        """
        Args:
            data (list of dicts): Each dict contains {'text': str, 'labels': dict}.
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length (int): Max token length for padding/truncation.
        """
        self.root = root
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.extract_data()

    @staticmethod
    def create_toy_dataset():
        '''
        Simulating some sample text data with bias attributes
        '''
        data = [{
            "source": "BBC Politics",
            "title": "International Development Minister Anneliese Dodds quits over aid cuts",
            "text": "International Development Minister Anneliese Dodds has resigned over the prime minister's cuts to the aid budget. In a letter to Sir Keir Starmer, Dodds said the cuts to international aid, announced earlier this week to fund an increase in defence spending, would \"remove food and healthcare from desperate people - deeply harming the UK's reputation\". She told the PM she had delayed her resignation until after his meeting with President Trump, saying it was \"imperative that you had a united cabinet behind you as you set off for Washington\". The Oxford East MP, who attended cabinet despite not being a cabinet minister, said it was with \"sadness\" that she was resigning. She said that while Sir Keir had been clear he was not \"ideologically opposed\" to international development, the cuts were \"being portrayed as following in President Trump's slipstream of cuts to USAID\". Ahead of his trip to meet the US president, Sir Keir announced aid funding would be reduced from 0.5% of gross national income to 0.3% in 2027 in order to fund an increase in defence spending. In his reply to Dodds's resignation letter, the prime minister thanked the departing minister for her \"hard work, deep commitment and friendship\". He said cutting aid was a \"difficult and painful decision and not one I take lightly\" adding: \"We will do everything we can...to rebuild a capability on development.\" In her resignation letter, Dodds said she welcomed an increase to defence spending at a time when the post-war global order had \"come crashing down\". She added that she understood some of the increase might have to be paid for by cuts to ODA [overseas development assistance]. However, she expressed disappointment that instead of discussing \"our fiscal rules and approach to taxation\", the prime minister had opted to allow the ODA to \"absorb the entire burden\". She said the cuts would \"likely lead to a UK pull-out from numerous African, Caribbean and Western Balkan nations - at a time when Russia has been aggressively increasing its global presence\". \"It will likely lead to withdrawal from regional banks and a reduced commitment to the World Bank; the UK being shut out of numerous multilateral bodies; and a reduced voice for the UK in the G7, G20 and in climate negotiations.\" The spending cuts mean \u00a36bn less will be spent on foreign aid each year. The aid budget is already used to pay for hotels for asylum seekers in the UK, meaning the actual amount spend on aid overseas will be around 0.15% of gross national income. The prime minister's decision to increase defence spending came ahead of his meeting in Washington - the US president has been critical of European countries for not spending enough on defence and instead relying on American military support. He welcomed the UK's commitment to spend more, but Sir Keir has been attacked by international development charities and some of his own MPs for the move. Dodds held off her announcement until the prime minister's return from Washington, in order not to overshadow the crucial visit, and it was clear she did not want to make things difficult for the prime minister. But other MPs have been uneasy about the decision, including Labour MP Sarah Champion, who chairs the international development committee, who said that cutting the aid budget to fund defence spending is a false economy that would \"only make the world less safe\". Labour MP Diane Abbott, who had been critical of the cuts earlier in the week, said it was \"shameful\" that other ministers had not resigned along with Dodds. Dodds's resignation also highlights that decisions the prime minister feels he has to take will be at odds with some of the views of Labour MPs, and those will add to tensions between the leadership and backbenchers. In a post on X, Conservative leader Kemi Badenoch said: \"I disagree with the PM on many things but on reducing the foreign aid budget to fund UK defence? He's absolutely right. \"He may not be able to convince the ministers in his own cabinet, but on this subject, I will back him.\" However one of her backbenchers - and a former international development minister - Andrew Mitchell backed Dodds, accusing Labour of trying \"disgraceful and cynical actions\". \"Shame on them and kudos to a politician of decency and principle,\" he added. Liberal Democrat international development spokesperson Monica Harding said Dodds had \"done the right thing\" describing the government's position as \"unsustainable. She said it was right to increase defence spending but added that \"doing so by cutting the international aid budget is like robbing Peter to pay Paul\". \"Where we withdraw our aid, it's Russia and China who will fill the vacuum.\" Deputy Prime Minister Angela Rayner said she was \"sorry to hear\" of Dodds's resignation. \"It is a really difficult decision that was made but it was absolutely right the PM and cabinet endorse the PM's actions to spend more money on defence,\" she said. Dodds first became a Labour MP in 2017 when she was elected to represent the Oxford East constituency. Under Jeremy Corbyn's leadership of the Labour Party she served as a shadow Treasury minister and was promoted to shadow chancellor when Sir Keir took over. Following Labour's poor performance in the 2021 local elections, she was demoted to the women and equalities brief. Since July 2024, she has served as international development minister. Dodds becomes the fourth minister to leave Starmer's government, following Louise Haigh, Tulip Siddiq and Andrew Gwynne. Some Labour MPs are unhappy about Tory defector Natalie Elphicke's past comments. The BBC chairman helped him secure a loan guarantee weeks before the then-PM recommended him for the role, a paper says. The minister is accused of using demeaning and intimidating language towards a civil servant when he was defence secretary. Eddie Reeves has been the Conservatives' leader on Oxfordshire County Council since May 2021. Labour chair Anneliese Dodds demands answers over \u00a3450,000 donation from ex-Tory treasurer Sir Ehud Sheleg. Copyright 2025 BBC. All rights reserved.\u00a0\u00a0The BBC is not responsible for the content of external sites.\u00a0Read about our approach to external linking. ",
            "url": "https://www.bbc.com/news/articles/cpv44982jlgo",
            "score": 43
        }]
        return data

    def extract_data(self):
        '''
        Extracts data from root folder
        '''
        if Path(self.root).exists():
            with open(self.root) as file:
                self.data = []
                datas = json.load(file)

                for idx, data in enumerate(datas):
                    encoding = self.tokenizer(
                        data['text'],
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    labels = {'bias_score': data['score'] / 100}
                    self.data.append({
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': torch.tensor(list(labels.values()), dtype=torch.float32)
                    })
        else:
            self.data = BiasDataset.create_toy_dataset()

    def __len__(self):
        '''
        Custom len function
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Custom getitem function
        '''
        item = self.data[idx]
        text = item['text']
        labels = {'bias_score': item['score']}  # Dictionary of multi-dimensional bias attributes
        
        # Tokenize text
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
