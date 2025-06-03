from ..data_basic import Dataset

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        if isinstance(i, slice):
            # Handle slice access - return batched data
            return tuple([a[i] for a in self.arrays])
        else:
            # Handle single index access
            return tuple([a[i] for a in self.arrays])