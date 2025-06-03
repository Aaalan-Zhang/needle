import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # Initialize iterator state
        self.batch_idx = 0
        
        if self.shuffle:
            # Create a new random ordering for this epoch
            indices = np.random.permutation(len(self.dataset))
            self.ordering = np.array_split(indices, 
                                         range(self.batch_size, len(self.dataset), self.batch_size))
        # If not shuffling, use the pre-computed ordering from __init__
        
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # Check if we've exhausted all batches
        if self.batch_idx >= len(self.ordering):
            raise StopIteration
        
        # Get the indices for the current batch
        batch_indices = self.ordering[self.batch_idx]
        self.batch_idx += 1
        
        # Fetch the data for this batch
        batch_items = []
        
        for idx in batch_indices:
            item = self.dataset[idx]
            batch_items.append(item)
        
        # Determine if we have labels by checking the first item
        if isinstance(batch_items[0], tuple):
            # Dataset returns tuples (data, label, ...)
            num_elements = len(batch_items[0])
            batched_elements = []
            
            for element_idx in range(num_elements):
                # Collect all elements at this position across the batch
                element_batch = [item[element_idx] for item in batch_items]
                # Convert to numpy array and then to Tensor
                element_array = np.array(element_batch)
                if element_idx == 0:  # Assume first element is data (float32)
                    element_tensor = Tensor(element_array, dtype="float32")
                else:  # Assume other elements are labels (uint8)
                    element_tensor = Tensor(element_array, dtype="uint8")
                batched_elements.append(element_tensor)
            
            return tuple(batched_elements)
        else:
            # Dataset returns single items (no tuples)
            batch_array = np.array(batch_items)
            batch_tensor = Tensor(batch_array, dtype="float32")
            return batch_tensor
        ### END YOUR SOLUTION

