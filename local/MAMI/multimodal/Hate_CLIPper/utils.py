import torch
import random
from torch.utils.data import Dataset
import emoji
import os
import copy
from PIL import Image
import numpy as np
from sklearn import metrics

def pad_clip_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    picel_values_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, picel_values, labls, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        picel_values_sequence.append(picel_values.squeeze(0))
        label_sequence.append(labls)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=49407)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    picel_values_sequence = torch.nn.utils.rnn.pad_sequence(picel_values_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, picel_values_sequence, label_sequence, filename_sequence

class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, val_file, test_file, tokenizer, device, max_len, task=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.task = task
        self.train_dataset, self.val_dataset, self.test_dataset= self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = CLIPdatasetloader(self.train_file, self.max_len, self.tokenizer, task=self.task)
        val_dataset = CLIPdatasetloader(self.val_file, self.max_len, self.tokenizer, task=self.task)
        test_dataset = CLIPdatasetloader(self.test_file, self.max_len, self.tokenizer, task=self.task)
        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len, tokenizer, task=None):
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.task = task
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        text = self.datadict[self.datakeys[index]]['text']
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir'])#.replace('./', './../../../'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        out = self.tokenizer(text=[text],
                            images=image,
                            return_tensors="pt",
                            padding=True)

        ids = out.data['input_ids']
        mask = out.data['attention_mask']
        pixel_values = out.data['pixel_values']
        filename = self.datakeys[index]
        
        if self.task == 'taskA':
            label = int(self.datadict[self.datakeys[index]]['taskA'][0])
            label = torch.FloatTensor([label])
        elif self.task == 'taskB':
            label = self.datadict[self.datakeys[index]]['taskB']
            label = [int(i) for i in label]
            label = torch.FloatTensor(label)
        

        if ids[0].size()[0] > self.max_len:
            newid = ids[0][:self.max_len].unsqueeze(0)
            newmask = mask[0][:self.max_len].unsqueeze(0)
            return newid, newmask, pixel_values, label, filename
        else:
            return ids, mask, pixel_values, label, filename  # twtfsingdata.squeeze(0), filename
