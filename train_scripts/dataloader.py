import torch
import json
from torch.utils.data import Dataset
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
    def __init__(self, root, tokenizer, analyzer=None, max_length=512):
        """
        Args:
            data (list of dicts): Each dict contains {'text': str, 'labels': dict}.
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length (int): Max token length for padding/truncation.
        """
        self.root = root
        self.tokenizer = tokenizer
        self.analyzer = analyzer
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
        },
        {
            "source": "BBC Politics",
            "title": "Donald Trump: UK-US trade deal could mean tariffs 'not necessary'",
            "text": "A trade deal between the US and UK could happen \"very quickly\", President Donald Trump said at a joint press conference with Sir Keir Starmer. Speaking during the prime minister's visit to the White House, Trump envisaged \"a real trade deal\" which could see the UK avoid the kind of tariffs the president has been threatening on some of the US's other trading partners. The trip had been seen as a key moment in Sir Keir's premiership as he sought to influence Trump's decisions on topics including Ukraine, as well as trade. Sir Keir kicked off his White House visit by presenting Trump with a letter from King Charles inviting him to an \"unprecedented\" second state visit to the UK. Receiving the letter in front of cameras in the Oval Office, Trump said it would be a \"great honour\" and described the King as \"a wonderful man\". Sir Keir said the offer of a second state visit was \"truly historic\". Traditionally US presidents have only been given one state visit. Having confirmed he would be accepting the invite, Trump, along with Sir Keir took questions from reporters for 30 minutes. The US president did most of the talking, setting out his stance on many subjects including the possibility of a Ukraine deal and the UK's potential agreement with Mauritius over the Chagos Islands. On the plane to the US, Sir Keir reiterated his willingness to send British troops to Ukraine as part of a peace deal. However, he argued that, without US security guarantees, Russian President Vladimir Putin could re-invade Ukraine. Asked if he would provide such assurances, Trump said a minerals agreement he plans to sign with Ukraine on Friday could provide a \"backstop\". He said \"nobody will play around\" if US workers were in the country, as part of the deal on minerals. The US president was pressed on whether he stood by his accusation that Ukrainian President Volodymyr Zelensky was a \"dictator\". \"Did I say that? I can't believe I said that,\" he said. He later added he had \"a lot of respect\" for Zelensky, who he will host in Washington DC on Friday. The UK's planned agreement with Mauritius over the Chagos Islands was one potential source of tension between the UK and US leaders. However, Trump appeared to back the UK's approach saying he was \"inclined to go along with it\". The deal would see the UK cede sovereignty of the Indian Ocean archipelago, but maintain control over the island of Diego Garcia, which includes a US-UK military airbase, by leasing it back. After taking questions in the Oval Office, the two leaders took part in talks and then held a formal press conference, during which Trump repeatedly spoke about a possible US-UK trade deal which could be agreed \"very quickly\". Referring to an economic, rather than a trade deal, Sir Keir said the UK and US  would begin work on an agreement centred on the potential of artificial intelligence. \"Instead of over-regulating these new technologies, we're seizing the opportunities they offer,\" he said. He said the UK and US had shaped the \"great technological innovations of the last century\" and now had the chance to do the same in the 21st Century. \"Artificial intelligence could cure cancer. That could be a moon shot for our age, and that's how we'll keep delivering for our people,\" he said. Trump has repeatedly threatened to impose tariffs - import taxes - on many of its allies, including 25% on goods made in the European Union. He also ordered a 25% import tax on all steel and aluminium entering the US - which could hit the UK. Asked if Sir Keir had tried to dissuade the president from ordering tariffs against the UK, Trump said: \"He tried.\" \"He was working hard I tell you that. He earned whatever the hell they pay him over there,\" he said. \"I think there's a very good chance that in the case of these two great, friendly countries, I think we could very well end up with a real trade deal where the tariffs wouldn't be necessary. We'll see.\" In a bid to convince the president against UK tariffs, Sir Keir said the US-UK trade relationship was \"fair, balanced and reciprocal\". Since leaving the European Union, successive British leaders have hoped to get a general free trade deal with the US. In his first term as president, Trump said talks about a \"very substantial\" trade deal with the UK were under way. However, negotiations stalled with disagreements over US agricultural exports and UK taxes on tech companies causing problems. The head of trade policy at the British Chambers of Commerce - a former Labour MP and minister - told BBC Radio 4's Today programme on Friday that businesses will be encouraged by what he called an \"important first step\". \"In trade negotiations, relationships matter,\" says William Bain, adding that seeing the two leaders find common ground on their respective economies and trade is \"helpful\". He added that a deal to keep tariffs low would most benefit automotive and pharmaceutical industries in the UK. Copyright 2025 BBC. All rights reserved.\u00a0\u00a0The BBC is not responsible for the content of external sites.\u00a0Read about our approach to external linking. ",
            "url": "https://www.bbc.com/news/articles/c7988r3q1p2o",
            "score": 50
        }]
        return data

    def extract_data(self):
        '''
        Extracts data from root folder
        '''
        if Path(self.root).exists():
            with open(self.root) as file:
                self.data = json.load(file)
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

        # Analayser
        if self.analyzer:
            analysis = self.analyzer(text)

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
