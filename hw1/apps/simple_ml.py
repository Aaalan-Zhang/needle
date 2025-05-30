"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        images = images.astype(np.float32) / 255.0

    with gzip.open(label_filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR CODE
    batch_size = Z.shape[0]
    
    # Compute log-sum-exp: log(sum(exp(Z), axis=1))
    # For numerical stability, we can subtract the max, but for simplicity let's compute directly
    exp_Z = ndl.exp(Z)  # exp of logits
    sum_exp_Z = ndl.summation(exp_Z, axes=(1,))  # sum over classes
    log_sum_exp = ndl.log(sum_exp_Z)  # log of sum
    
    # Compute the true class logits: sum(Z * y_one_hot, axis=1)
    true_class_logits = ndl.summation(Z * y_one_hot, axes=(1,))
    
    # Softmax loss = log_sum_exp - true_class_logits
    loss_per_sample = log_sum_exp - true_class_logits
    
    # Return average loss over the batch
    total_loss = ndl.summation(loss_per_sample)
    return total_loss / batch_size
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    
    for i in range(0, num_examples, batch):
        # Get batch data
        end_idx = min(i + batch, num_examples)
        Xb = X[i:end_idx]
        yb = y[i:end_idx]
        batch_size = Xb.shape[0]
        
        # Convert batch to needle tensors
        Xb_tensor = ndl.Tensor(Xb)
        
        # Create one-hot encoded labels
        y_one_hot = np.zeros((batch_size, W2.shape[1]))
        y_one_hot[np.arange(batch_size), yb] = 1
        y_one_hot_tensor = ndl.Tensor(y_one_hot)
        
        # Forward pass
        Z1 = ndl.relu(ndl.matmul(Xb_tensor, W1))  # ReLU(X * W1)
        logits = ndl.matmul(Z1, W2)               # ReLU(X * W1) * W2
        
        # Compute loss using our softmax_loss function
        loss = softmax_loss(logits, y_one_hot_tensor)
        
        # Backward pass - compute gradients
        loss.backward()
        
        # Update weights using computed gradients
        W1_data = W1.realize_cached_data() - lr * W1.grad.realize_cached_data()
        W2_data = W2.realize_cached_data() - lr * W2.grad.realize_cached_data()
        
        # Create new weight tensors (detached from computation graph)
        W1 = ndl.Tensor(W1_data, requires_grad=True)
        W2 = ndl.Tensor(W2_data, requires_grad=True)
    
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
