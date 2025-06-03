"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # Initialize weight first, then bias (as specified in homework)
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, 
            device=device, dtype=dtype, requires_grad=True
        ))
        
        if bias:
            # Create bias with fan_in = out_features as specified in homework
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 1, 
                device=device, dtype=dtype, requires_grad=True
            ).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # y = xA^T + b
        out = X @ self.weight
        if self.bias is not None:
            # Explicitly broadcast bias to correct shape
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # Flatten all non-batch dimensions: (B, X_0, X_1, ...) -> (B, X_0 * X_1 * ...)
        batch_size = X.shape[0]
        flat_size = 1
        for dim in X.shape[1:]:
            flat_size *= dim
        return X.reshape((batch_size, flat_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Apply modules sequentially
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # ℓ_softmax(z,y) = log Σ exp(z_i) - z_y
        # Use logsumexp for numerical stability
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Convert y to one-hot encoding
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        
        # Compute log sum exp over classes (axis=1)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        
        # Get the logits for true labels: z_y
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        
        # Loss = log_sum_exp - z_y, then take mean over batch
        loss = log_sum_exp - z_y
        return ops.summation(loss) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Initialize learnable parameters
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        
        # Initialize running statistics (not parameters)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Training mode: use batch statistics
            # Compute mean and variance over batch dimension (axis=0)
            batch_mean = ops.summation(x, axes=(0,)) / x.shape[0]
            
            # Compute variance: E[(x - μ)²]
            x_centered = x - batch_mean.broadcast_to(x.shape)
            batch_var = ops.summation(x_centered ** 2, axes=(0,)) / x.shape[0]
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            # Normalize using batch statistics
            norm = (x - batch_mean.broadcast_to(x.shape)) / ((batch_var + self.eps).broadcast_to(x.shape) ** 0.5)
        else:
            # Evaluation mode: use running statistics
            norm = (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var + self.eps).broadcast_to(x.shape) ** 0.5)
        
        # Apply learnable transformation
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Initialize learnable parameters
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Compute mean and variance across features (axis=1) for each sample
        feature_mean = ops.summation(x, axes=(1,)) / x.shape[1]
        
        # Reshape mean to broadcast properly: (batch_size, 1)
        feature_mean = feature_mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        # Compute variance
        x_centered = x - feature_mean
        feature_var = ops.summation(x_centered ** 2, axes=(1,)) / x.shape[1]
        feature_var = feature_var.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        # Normalize
        norm = x_centered / ((feature_var + self.eps) ** 0.5)
        
        # Apply learnable transformation
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # During training: randomly zero elements with probability p
            # Scale by 1/(1-p) to maintain expected value
            mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            # During evaluation: identity function
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Return F(x) + x
        return self.fn(x) + x
        ### END YOUR SOLUTION
