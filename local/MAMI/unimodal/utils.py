import torch
from torch.utils.data import Dataset
from PIL import Image

def pad_clip_custom_sequence(sequences):
    """
    Custom collate function to pad CLIP sequences for batching.

    Parameters
    ----------
    sequences : Sequence of tuples
        Each tuple contains:
        - node_sets (torch.Tensor)
        - mask (torch.Tensor)
        - pixel_values (torch.Tensor)
        - label (torch.Tensor)
        - filename (str)

    Returns
    -------
    tuple
        - node_sets_sequence : torch.Tensor
            Padded tensor of node sets.
        - mask_sequence : torch.Tensor
            Padded tensor of masks.
        - pixel_values_sequence : torch.Tensor
            Tensor of pixel values.
        - label_sequence : torch.Tensor
            Tensor of labels.
        - filename_sequence : list of str
            List of filenames corresponding to the batch.
    """
    node_sets_sequence = []
    mask_sequence = []
    pixel_values_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, picel_values, labls, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        pixel_values_sequence.append(picel_values.squeeze(0))
        label_sequence.append(labls)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=49407)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    pixel_values_sequence = torch.nn.utils.rnn.pad_sequence(pixel_values_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, pixel_values_sequence, label_sequence, filename_sequence

class CLIPdatasetclass:  
    """
    Construct train, validation, and test datasets for CLIP-based tasks.
    """
    def __init__(self, train_file, val_file, test_file, tokenizer, device, max_len, modal, task=None):
        """
        Initialize the dataset class.

        Parameters
        ----------
        train_file : dict
            Dictionary of the training set.
        val_file : dict
            Dictionary of the validation set.
        test_file : dict
            Dictionary of the test set.
        tokenizer : object
            Tokenizer used for text preprocessing.
        device : str
            Device identifier (e.g., 'cpu' or 'cuda').
        max_len : int
            Maximum sequence length.
        modal : str
            Modality used ('mami_text' or 'mami_image').
        task : str, optional
            Task identifier (e.g., 'taskA' or 'taskB').
        """
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.task = task
        self.modal = modal
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_dataset()


    def prepare_dataset(self):  
        """
        Prepare train, validation, and test datasets.

        Returns
        -------
        tuple of datasets
            (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = CLIPdatasetloader(self.train_file, self.max_len, self.tokenizer, self.modal, task=self.task)
        val_dataset = CLIPdatasetloader(self.val_file, self.max_len, self.tokenizer, self.modal, task=self.task)
        test_dataset = CLIPdatasetloader(self.test_file, self.max_len, self.tokenizer, self.modal, task=self.task)
        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset): 
    """
    Dataset class for creating train, validation, and test sets for CLIP-based tasks.
    """
    def __init__(self, datadict, max_len, tokenizer, modal, task=None):
        """
        Initialize dataset loader.

        Parameters
        ----------
        datadict : dict
            Dictionary mapping filenames to their text, image paths, and labels.
        max_len : int
            Maximum token sequence length.
        tokenizer : object
            Tokenizer for processing text and images.
        modal : str
            Modality used ('mami_text' or 'mami_image').
        task : str, optional
            Task identifier (e.g., 'taskA' or 'taskB').
        """
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.task = task
        self.modal = modal
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_keys(self, datadict):
        """
        Return all keys in the dataset dictionary.

        Parameters
        ----------
        datadict : dict
            Dataset dictionary.

        Returns
        -------
        list of str
            Keys corresponding to dataset entries.
        """
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.datakeys)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple
            (ids, mask, pixel_values, label, filename)
        """

        # load text data
        text = self.datadict[self.datakeys[index]]['text']

        # load image data
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir']) 
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # toknize text and image into tokens
        out = self.tokenizer(text=[text],
                            images=image,
                            return_tensors="pt",
                            max_length = self.max_len,
                            padding='max_length')

        ids = out.data['input_ids']
        mask = out.data['attention_mask']
        pixel_values = out.data['pixel_values']
        filename = self.datakeys[index]

        # load ground truth for tasks
        if self.task == 'taskA':
            label = int(self.datadict[self.datakeys[index]]['taskA'])
            label = torch.FloatTensor([label])
        elif self.task == 'taskB':
            label = self.datadict[self.datakeys[index]]['taskB']
            label = [int(i) for i in label]
            label = torch.FloatTensor(label)
        
        # truncated the token and masks due to the maximum token length limitation
        if ids[0].size()[0] > self.max_len:
            newid = ids[0][:self.max_len].unsqueeze(0)
            newmask = mask[0][:self.max_len].unsqueeze(0)
            return newid, newmask, pixel_values, label, filename
        else:
            return ids, mask, pixel_values, label, filename  