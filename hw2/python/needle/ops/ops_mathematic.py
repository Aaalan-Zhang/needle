"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # d/dx(x^y) = y * x^(y-1) * dx + ln(x) * x^y * dy
        lhs_grad = out_grad * rhs * power(lhs, rhs - 1)
        rhs_grad = out_grad * log(lhs) * power(lhs, rhs)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        # d/dx(x^c) = c * x^(c-1)
        return (out_grad * self.scalar * power_scalar(input_tensor, self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # d/dx(x/y) = 1/y * dx + (-x/y^2) * dy
        lhs_grad = out_grad / rhs
        rhs_grad = out_grad * (-lhs) / (rhs * rhs)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d/dx(x/c) = 1/c
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # Default: transpose last two dimensions
            ndim = len(a.shape)
            if ndim < 2:
                return a
            axes = list(range(ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return array_api.transpose(a, axes)
        else:
            # axes should be a permutation of all dimensions
            if len(self.axes) == 2:
                # If only two axes given, assume we want to swap those two dimensions
                ndim = len(a.shape)
                axes = list(range(ndim))
                axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
                return array_api.transpose(a, axes)
            else:
                # Full permutation given
                return array_api.transpose(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of transpose is transpose with inverted axes
        if self.axes is None:
            # Default case: transpose last two dimensions again
            return transpose(out_grad, self.axes)
        else:
            if len(self.axes) == 2:
                # If we swapped two dimensions, swap them back
                return transpose(out_grad, self.axes)
            else:
                # Invert the full permutation
                inv_axes = [0] * len(self.axes)
                for i, axis in enumerate(self.axes):
                    inv_axes[axis] = i
                return transpose(out_grad, tuple(inv_axes))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of reshape is reshape back to original shape
        input_tensor = node.inputs[0]
        return reshape(out_grad, input_tensor.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of broadcast is sum over broadcasted dimensions
        input_tensor = node.inputs[0]
        input_shape = input_tensor.shape
        
        # Sum over dimensions that were broadcasted
        grad = out_grad
        
        # Handle case where input has fewer dimensions than output
        ndims_added = len(self.shape) - len(input_shape)
        for _ in range(ndims_added):
            grad = summation(grad, axes=(0,))
        
        # Handle case where input dimensions were size 1 and got broadcasted
        for i, (input_dim, output_dim) in enumerate(zip(input_shape, self.shape[-len(input_shape):])):
            if input_dim == 1 and output_dim > 1:
                grad = summation(grad, axes=(i,))
                grad = reshape(grad, grad.shape[:i] + (1,) + grad.shape[i:])
        
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Gradient of sum is broadcast to original shape
        input_tensor = node.inputs[0]
        input_shape = input_tensor.shape
        
        # Reshape out_grad to match the shape after summation but before broadcasting
        if self.axes is None:
            # Sum over all axes results in scalar, broadcast to original shape
            grad = broadcast_to(reshape(out_grad, (1,) * len(input_shape)), input_shape)
        else:
            # Sum over specific axes
            # Insert dimensions of size 1 where summation happened
            grad_shape = list(out_grad.shape)
            if isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            
            for axis in sorted(axes):
                if axis < 0:
                    axis = len(input_shape) + axis
                grad_shape.insert(axis, 1)
            
            grad = reshape(out_grad, grad_shape)
            grad = broadcast_to(grad, input_shape)
        
        return grad
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # d/dA(AB) = out_grad @ B^T
        # d/dB(AB) = A^T @ out_grad
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        
        # Handle broadcasting: if input shapes differ from gradient shapes,
        # we need to sum over the broadcasted dimensions
        
        # For lhs_grad: sum over extra batch dimensions if lhs had fewer dims
        if len(lhs_grad.shape) > len(lhs.shape):
            # Sum over leading dimensions that were broadcasted
            axes_to_sum = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            lhs_grad = summation(lhs_grad, axes=axes_to_sum)
        
        # For rhs_grad: sum over extra batch dimensions if rhs had fewer dims  
        if len(rhs_grad.shape) > len(rhs.shape):
            # Sum over leading dimensions that were broadcasted
            axes_to_sum = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            rhs_grad = summation(rhs_grad, axes=axes_to_sum)
        
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d/dx(-x) = -1
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        # d/dx(ln(x)) = 1/x
        return out_grad / input_tensor
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        # d/dx(e^x) = e^x
        return out_grad * exp(input_tensor)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        # d/dx(ReLU(x)) = 1 if x > 0, 0 if x <= 0
        # Create a mask where input > 0
        mask = Tensor(input_tensor.realize_cached_data() > 0, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

