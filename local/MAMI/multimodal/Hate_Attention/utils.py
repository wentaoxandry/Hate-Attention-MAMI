import torch
import random
from torch.utils.data import Dataset
import emoji
import os
import copy
from PIL import Image
import numpy as np
from sklearn import metrics

class Feature_extractor:  
    def __init__(self, image_processor, text_processor, task):
        self.image_processor = image_processor  
        self.text_processor = text_processor  
        self.task = task
    def __call__(self, batch):
        texts = self.text_processor([item[0] for item in batch], padding=True, return_tensors="pt", truncation=True)
        #tags = self.text_processor([item[1] for item in batch], padding=True, return_tensors="pt", truncation=True)
        pixel_values = self.image_processor(images=[item[1] for item in batch], return_tensors="pt")["pixel_values"]
        if self.task == 'taskA':
            labels = torch.FloatTensor([item[2] for item in batch]).unsqueeze(-1)
        else:
            labels = torch.FloatTensor([item[2] for item in batch])
        filename = [item[3] for item in batch]
        input_text_ids = texts['input_ids']
        attention_text_masks = texts['attention_mask']

        return input_text_ids, attention_text_masks, pixel_values, labels, filename

class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, val_file, test_file, device, max_len, task=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.device = device
        self.max_len = max_len
        self.task = task
        self.train_dataset, self.val_dataset, self.test_dataset= self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = CLIPdatasetloader(self.train_file, self.max_len, task=self.task)
        val_dataset = CLIPdatasetloader(self.val_file, self.max_len, task=self.task)
        test_dataset = CLIPdatasetloader(self.test_file, self.max_len, task=self.task)
        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len, task=None):
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.task = task
        self.max_len = max_len

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        #text = normalizeTweet(self.datadict[self.datakeys[index]]['text'].replace('\n', ' '))
        text = self.datadict[self.datakeys[index]]['text']
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir'])#.replace('./', './../../../'))
        image = image.convert('RGB')
        filename = self.datakeys[index]
        if self.task == 'taskA':
            label = int(self.datadict[self.datakeys[index]]['taskA'][0])
            #label = torch.FloatTensor([label])
        elif self.task == 'taskB':
            label = self.datadict[self.datakeys[index]]['taskB']
            label = [int(i) for i in label]
            #label = torch.FloatTensor(label)
   
        return text, image, label, filename

