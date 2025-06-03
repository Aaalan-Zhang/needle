from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR CODE
        super().__init__(transforms)
        
        # Load images
        with gzip.open(image_filename, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
            images = images.astype(np.float32) / 255.0

        # Load labels
        with gzip.open(label_filename, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Store as instance attributes
        self.images = images
        self.labels = labels
        self.num_rows = num_rows
        self.num_cols = num_cols
        ### END YOUR CODE

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, slice):
            # Handle slice access - return batched data
            start, stop, step = index.indices(len(self))
            indices = list(range(start, stop, step))
            
            # Collect all data and labels for the slice
            batch_images = []
            batch_labels = []
            
            for idx in indices:
                image = self.images[idx].reshape(self.num_rows, self.num_cols, 1)
                image = self.apply_transforms(image)
                batch_images.append(image)
                batch_labels.append(self.labels[idx])
            
            # Stack into arrays
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            
            return batch_images, batch_labels
        else:
            # Handle single index access
            # Get the image and reshape it to H x W x C format (28 x 28 x 1 for MNIST)
            image = self.images[index].reshape(self.num_rows, self.num_cols, 1)
            label = self.labels[index]
            
            # Apply transforms if any
            image = self.apply_transforms(image)
            
            return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION