import torch
from torch.utils.data import Dataset
from PIL import Image

class Feature_extractor:  
    """
    Collate/feature-extraction helper that tokenizes text, preprocesses images,
    and assembles labels/filenames for batching.
    """
    def __init__(self, image_processor, text_processor, task):
        """
        Initialize the feature extractor.

        Parameters
        ----------
        image_processor : object
            Image preprocessor (e.g., a CLIP image processor) that returns a dict
            containing a "pixel_values" tensor when called.
        text_processor : object
            Text tokenizer/processor (e.g., a CLIP text tokenizer) that returns a dict
            with 'input_ids' and 'attention_mask' tensors when called.
        task : str
            Task identifier, e.g., 'taskA' (binary) or 'taskB' (multi-label).
        """
        self.image_processor = image_processor  
        self.text_processor = text_processor  
        self.task = task
    def __call__(self, batch):
        """
        Process a batch of raw samples into model-ready tensors.

        Parameters
        ----------
        batch : list of tuple
            Each item is a 4-tuple: (text: str, image: object, label: int|list[int], filename: str).

        Returns
        -------
        tuple
            input_text_ids : torch.Tensor
                Token IDs for the text encoder, shape (batch_size, seq_len).
            attention_text_masks : torch.Tensor
                Attention masks for the text encoder, shape (batch_size, seq_len).
            pixel_values : torch.Tensor
                Preprocessed image tensor, typically (batch_size, 3, H, W).
            labels : torch.Tensor
                Labels tensor; shape (batch_size, 1) for taskA, (batch_size, 4) for taskB.
            filename : list of str
                Filenames corresponding to each batch item.
        """
        texts = self.text_processor([item[0] for item in batch], padding=True, return_tensors="pt", truncation=True)
        pixel_values = self.image_processor(images=[item[1] for item in batch], return_tensors="pt")["pixel_values"]
        if self.task == 'taskA':
            labels = torch.FloatTensor([item[2] for item in batch]).unsqueeze(-1)
        else:
            labels = torch.FloatTensor([item[2] for item in batch])
        filename = [item[3] for item in batch]
        input_text_ids = texts['input_ids']
        attention_text_masks = texts['attention_mask']

        return input_text_ids, attention_text_masks, pixel_values, labels, filename

class CLIPdatasetclass:  
    """
    Construct train, validation, and test datasets for CLIP-based tasks.
    """
    def __init__(self, train_file, val_file, test_file, device, task=None):
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
        device : str
            Device identifier (e.g., 'cpu' or 'cuda').
        task : str, optional
            Task identifier (e.g., 'taskA' or 'taskB').
        """
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.device = device
        self.task = task
        self.train_dataset, self.val_dataset, self.test_dataset= self.prepare_dataset()


    def prepare_dataset(self):  
        """
        Prepare train, validation, and test datasets.

        Returns
        -------
        tuple of datasets
            (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = CLIPdatasetloader(self.train_file, task=self.task)
        val_dataset = CLIPdatasetloader(self.val_file, task=self.task)
        test_dataset = CLIPdatasetloader(self.test_file, task=self.task)
        return train_dataset, val_dataset, test_dataset


class CLIPdatasetloader(Dataset): 
    """
    Dataset class for creating train, validation, and test sets for CLIP-based tasks.
    """
    def __init__(self, datadict, task=None):
        """
        Initialize dataset loader.

        Parameters
        ----------
        datadict : dict
            Dictionary mapping filenames to their text, image paths, and labels.
        task : str, optional
            Task identifier (e.g., 'taskA' or 'taskB').
        """
        super(CLIPdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.task = task

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
            (text, image, label, filename)
        """

        # load text data
        text = self.datadict[self.datakeys[index]]['text']

        # load image data
        image = Image.open(self.datadict[self.datakeys[index]]['imagedir'])
        image = image.convert('RGB')

        filename = self.datakeys[index]

        # load ground truth for tasks
        if self.task == 'taskA':
            label = int(self.datadict[self.datakeys[index]]['taskA'][0])
        elif self.task == 'taskB':
            label = self.datadict[self.datakeys[index]]['taskB']
            label = [int(i) for i in label]

   
        return text, image, label, filename

