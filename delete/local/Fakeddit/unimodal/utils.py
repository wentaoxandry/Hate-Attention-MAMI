import torch
import random
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn import metrics
from PIL import Image, ImageFile
import kaldiio
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _get_from_loader(filepath, filetype):
    """Return ndarray

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    #>>> ndarray = loader.get_from_loader(
    #...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filepath:
    :param: str filetype:
    :return:
    :rtype: np.ndarray
    """
    if filetype in ['mat', 'vec']:
        # e.g.
        #    {"input": [{"feat": "some/path.ark:123",
        #                "filetype": "mat"}]},
        # In this case, "123" indicates the starting points of the matrix
        # load_mat can load both matrix and vector
        filepath = filepath
        return kaldiio.load_mat(filepath)
    elif filetype == 'scp':
        # e.g.
        #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
        #                "filetype": "scp",
        filepath, key = filepath.split(':', 1)
        loader = self._loaders.get(filepath)
        if loader is None:
            # To avoid disk access, create loader only for the first time
            loader = kaldiio.load_scp(filepath)
            self._loaders[filepath] = loader
        return loader[key]
    else:
        raise NotImplementedError(
            'Not supported: loader_type={}'.format(filetype))
    
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
    text_sequence = []
    image_sequence = []
    label_sequence = []
    filename_sequence = []
    for textfeat, imagefeat, labls, filename in sequences:
        text_sequence.append(textfeat.squeeze(0))
        image_sequence.append(imagefeat.squeeze(0))
        label_sequence.append(labls)
        filename_sequence.append(filename)
    text_sequence = torch.nn.utils.rnn.pad_sequence(text_sequence, batch_first=True)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return text_sequence, image_sequence, label_sequence.squeeze(-1), filename_sequence

class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, val_file, test_file, modal, max_len):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.modal = modal
        self.max_len = max_len
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask

        train_dataset = CLIPdatasetloader(self.train_file, self.max_len)
        val_dataset = CLIPdatasetloader(self.val_file, self.max_len)
        test_dataset = CLIPdatasetloader(self.test_file, self.max_len)


        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len):
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.max_len = max_len

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        textfeatdir = self.datadict[self.datakeys[index]]['textfeat']
        textfeat = _get_from_loader(
            filepath=textfeatdir,
            filetype='mat')
        imagefeatdir = self.datadict[self.datakeys[index]]['imagefeat']
        imagefeat = _get_from_loader(
            filepath=imagefeatdir,
            filetype='mat')
        textfeat = torch.FloatTensor(textfeat)
        imagefeat = torch.FloatTensor(imagefeat)
        filename = self.datakeys[index]
        label = int(self.datadict[self.datakeys[index]]['6_way_label'])
        label = torch.LongTensor([label])

        return textfeat, imagefeat, label, filename

