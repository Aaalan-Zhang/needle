from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # LogSoftmax(z) = z - LogSumExp(z)
        # Compute over the last axis (axis=-1) for typical use
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        shifted_z = Z - max_z
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(shifted_z), axis=-1, keepdims=True)) + max_z
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d/dz LogSoftmax(z) = I - softmax(z)
        # where softmax(z) = exp(LogSoftmax(z))
        log_softmax_out = node
        softmax_out = exp(log_softmax_out)
        # Gradient: out_grad * (I - softmax)
        # This is: out_grad - softmax * sum(out_grad, axis=-1, keepdims=True)
        sum_out_grad = summation(out_grad, axes=(-1,)).reshape(out_grad.shape[:-1] + (1,)).broadcast_to(out_grad.shape)
        return out_grad - softmax_out * sum_out_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # LogSumExp(z) = log(Σ exp(z_i - max z)) + max z
        if self.axes is None:
            # Reduce over all axes
            max_z = array_api.max(Z)
            shifted_z = Z - max_z
            return array_api.log(array_api.sum(array_api.exp(shifted_z))) + max_z
        else:
            # Reduce over specified axes
            max_z = array_api.max(Z, axis=self.axes, keepdims=True)
            shifted_z = Z - max_z
            log_sum_exp = array_api.log(array_api.sum(array_api.exp(shifted_z), axis=self.axes, keepdims=True)) + max_z
            # Remove the keepdims if needed
            return array_api.squeeze(log_sum_exp, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d/dz LogSumExp(z) = exp(z - LogSumExp(z)) = softmax(z)
        input_val = node.inputs[0]
        
        # Compute softmax: exp(z - logsumexp(z))
        if self.axes is None:
            # If reducing over all axes
            log_sum_exp_expanded = node.broadcast_to(input_val.shape)
        else:
            # Need to expand logsumexp result to match input shape
            # First, figure out the shape after reduction
            reduced_shape = list(input_val.shape)
            if isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            
            # Insert dimensions back
            node_expanded = node
            for axis in sorted(axes):
                node_expanded = node_expanded.reshape(
                    node_expanded.shape[:axis] + (1,) + node_expanded.shape[axis:]
                )
            log_sum_exp_expanded = node_expanded.broadcast_to(input_val.shape)
        
        softmax = exp(input_val - log_sum_exp_expanded)
        
        # Gradient is out_grad broadcasted and multiplied by softmax
        if self.axes is None:
            out_grad_expanded = out_grad.broadcast_to(input_val.shape)
        else:
            # Need to expand out_grad to match input shape
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            out_grad_expanded = out_grad
            for axis in sorted(axes):
                out_grad_expanded = out_grad_expanded.reshape(
                    out_grad_expanded.shape[:axis] + (1,) + out_grad_expanded.shape[axis:]
                )
            out_grad_expanded = out_grad_expanded.broadcast_to(input_val.shape)
        
        return out_grad_expanded * softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

