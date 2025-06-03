import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # A residual block consists of:
    # Linear -> Norm -> ReLU -> Dropout -> Linear -> Norm -> ReLU -> Dropout
    # with a residual connection around the whole block
    
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # MLPResNet consists of:
    # 1. Initial linear layer to transform input to hidden_dim
    # 2. Multiple residual blocks
    # 3. Final linear layer for classification
    
    layers = []
    
    # Initial transformation to hidden dimension
    layers.append(nn.Linear(dim, hidden_dim))
    layers.append(nn.ReLU())
    
    # Add residual blocks
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    
    # Final classification layer
    layers.append(nn.Linear(hidden_dim, num_classes))
    
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0
    
    # Set model to training mode if optimizer is provided, eval mode otherwise
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    loss_fn = nn.SoftmaxLoss()
    
    for batch in dataloader:
        X, y = batch
        
        # Flatten the input data (from (batch_size, H, W, C) to (batch_size, H*W*C))
        batch_size = X.shape[0]
        X_flat = X.reshape((batch_size, -1))
        
        # Forward pass
        logits = model(X_flat)
        loss = loss_fn(logits, y)
        
        # Calculate error rate (number of incorrect predictions)
        predictions = logits.numpy().argmax(axis=1)
        targets = y.numpy()
        error = (predictions != targets).sum()
        
        # Update totals
        total_loss += loss.numpy() * batch_size
        total_error += error
        total_samples += batch_size
        
        # Backward pass and optimization (only if training)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    # Return average error rate and average loss
    avg_error = total_error / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_error, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Create datasets
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", 
        f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    
    # Create data loaders
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model (MNIST images are 28x28 = 784 pixels)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    
    # Create optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch_num in range(epochs):
        # Training epoch
        train_err, train_loss = epoch(train_dataloader, model, opt)
        
        # Evaluation epoch
        test_err, test_loss = epoch(test_dataloader, model, None)
        
        print(f"Epoch {epoch_num}: Train err: {train_err:.4f}, Train loss: {train_loss:.4f}, "
              f"Test err: {test_err:.4f}, Test loss: {test_loss:.4f}")
    
    # Return only the final values from the last epoch
    return (train_err, train_loss, test_err, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
