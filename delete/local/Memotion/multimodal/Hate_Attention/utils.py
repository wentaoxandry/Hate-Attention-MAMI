import torch
import random
from torch.utils.data import Dataset
import os
import math
import copy
from PIL import Image
import numpy as np
from sklearn import metrics

class Feature_extractor:
    def __init__(self, image_processor, text_processor):
        self.image_processor = image_processor
        self.text_processor = text_processor
    def __call__(self, batch):
        texts = self.text_processor([item[0] for item in batch], padding=True, return_tensors="pt", truncation=True)
        pixel_values = self.image_processor(images=[item[1] for item in batch], return_tensors="pt")["pixel_values"]
        text_logits = torch.FloatTensor([item[2] for item in batch])
        image_logits = torch.FloatTensor([item[3] for item in batch])
        score = torch.FloatTensor([item[4] for item in batch])
        labelA = torch.LongTensor([item[5] for item in batch])
        labelB1 = torch.LongTensor([item[6] for item in batch])
        labelB2 = torch.LongTensor([item[7] for item in batch])
        labelB3 = torch.LongTensor([item[8] for item in batch])
        labelB4 = torch.LongTensor([item[9] for item in batch])
        filename = [item[10] for item in batch]
        input_text_ids = texts['input_ids']
        attention_text_masks = texts['attention_mask']

        return (input_text_ids, attention_text_masks, pixel_values, text_logits,
                image_logits, score, labelA, labelB1, labelB2, labelB3, labelB4, filename)
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
    labelA_sequence = []
    labelB1_sequence = []
    labelB2_sequence = []
    labelB3_sequence = []
    labelB4_sequence = []
    filename_sequence = []
    for node_sets, mask, picel_values, labelA, labelB1, labelB2, labelB3, labelB4, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        picel_values_sequence.append(picel_values.squeeze(0))
        labelA_sequence.append(labelA)
        labelB1_sequence.append(labelB1)
        labelB2_sequence.append(labelB2)
        labelB3_sequence.append(labelB3)
        labelB4_sequence.append(labelB4)

        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=49407)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    picel_values_sequence = torch.nn.utils.rnn.pad_sequence(picel_values_sequence, batch_first=True)
    labelA_sequence = torch.nn.utils.rnn.pad_sequence(labelA_sequence, batch_first=True)
    labelB1_sequence = torch.nn.utils.rnn.pad_sequence(labelB1_sequence, batch_first=True)
    labelB2_sequence = torch.nn.utils.rnn.pad_sequence(labelB2_sequence, batch_first=True)
    labelB3_sequence = torch.nn.utils.rnn.pad_sequence(labelB3_sequence, batch_first=True)
    labelB4_sequence = torch.nn.utils.rnn.pad_sequence(labelB4_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, picel_values_sequence, labelA_sequence.squeeze(-1), labelB1_sequence.squeeze(-1), \
            labelB2_sequence.squeeze(-1), labelB3_sequence.squeeze(-1), labelB4_sequence.squeeze(-1), filename_sequence
class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, val_file, test_file, tokenizer, device, max_len, task=None):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.task = task
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_dataset()


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
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir'])#.replace('./', './../../'))
        image = image.convert('RGB')
        filename = self.datakeys[index]
        out = self.tokenizer(text=[text],
                            images=image,
                            return_tensors="pt",
                            max_length = self.max_len,
                            padding='max_length')
        ids = out.data['input_ids']
        mask = out.data['attention_mask']
        pixel_values = out.data['pixel_values']
        labelA = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskA'])])
        labelB1 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['humorous'])])
        labelB2 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['sarcastic'])])
        labelB3 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['offensive'])])
        labelB4 = torch.LongTensor([int(self.datadict[self.datakeys[index]]['taskC']['motivation'])])
        if ids[0].size()[0] > self.max_len:
            newid = ids[0][:self.max_len].unsqueeze(0)
            newmask = mask[0][:self.max_len].unsqueeze(0)
            return newid, newmask, pixel_values, labelA, labelB1, labelB2, labelB3, labelB4, filename
        else:
            return ids, mask, pixel_values, labelA, labelB1, labelB2, labelB3, labelB4, filename
