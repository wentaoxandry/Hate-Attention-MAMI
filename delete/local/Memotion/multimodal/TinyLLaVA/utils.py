import torch
from torch.utils.data import Dataset
import emoji
import os
import copy
from PIL import Image
import numpy as np
from sklearn import metrics

class Feature_extractor:  
    def __init__(self, processor, task):
        self.processor = processor  
        self.task = task
    def __call__(self, batch):
        text = [item[0] for item in batch]
        image = [item[1] for item in batch]
        inputs = self.processor(text=text, images=image, padding=True, return_tensors="pt", truncation=True)
        
        labelA = torch.LongTensor([item[2] for item in batch])
        labelB1 = torch.LongTensor([item[3] for item in batch])
        labelB2 = torch.LongTensor([item[4] for item in batch])
        labelB3 = torch.LongTensor([item[5] for item in batch])
        labelB4 = torch.LongTensor([item[6] for item in batch])
        filename = [item[7] for item in batch]
    
        return inputs, labelA, labelB1, labelB2, labelB3, labelB4, filename

class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, val_file, test_file, device, task=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.device = device
        self.task = task
        self.train_dataset, self.val_dataset, self.test_dataset= self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = CLIPdatasetloader(self.train_file, task=self.task)
        val_dataset = CLIPdatasetloader(self.val_file, task=self.task)
        test_dataset = CLIPdatasetloader(self.test_file, task=self.task)
        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, task=None):
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.task = task

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        #text = normalizeTweet(self.datadict[self.datakeys[index]]['text'].replace('\n', ' '))
        text = self.datadict[self.datakeys[index]]['text']
        text = "USER: <image>\n" + text
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir'])#.replace('./', './../../../'))
        image = image.convert('RGB')
        filename = self.datakeys[index]
        labelA = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskA'])])
        labelB1 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['humorous'])])
        labelB2 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['sarcastic'])])
        labelB3 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['offensive'])])
        labelB4 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['motivation'])])
   
        return text, image, labelA, labelB1, labelB2, labelB3, labelB4, filename

