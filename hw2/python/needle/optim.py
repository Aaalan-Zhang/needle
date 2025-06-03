"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
                
            # Get gradient and apply weight decay if specified
            grad = param.grad.data
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Initialize momentum term if not exists
            if param not in self.u:
                self.u[param] = ndl.zeros_like(param.data)
            
            # Update momentum: u_{t+1} = β u_t + (1-β) ∇f(θ_t)
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            
            # Update parameter: θ_{t+1} = θ_t - α u_{t+1}
            # Ensure the result maintains the same dtype as the parameter
            update = self.lr * self.u[param]
            param.data = ndl.Tensor(param.data.numpy() - update.numpy(), 
                                   dtype=param.dtype, device=param.device)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # Compute total norm of all gradients
        total_norm = 0
        for param in self.params:
            if param.grad is not None:
                param_norm = (param.grad.data ** 2).sum()
                total_norm += param_norm
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for param in self.params:
                if param.grad is not None:
                    param.grad = param.grad * clip_coef
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1  # Increment time step
        
        for param in self.params:
            if param.grad is None:
                continue
                
            # Get gradient and apply weight decay if specified
            grad = param.grad.data
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Initialize first and second moment estimates if not exists
            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.data)
                self.v[param] = ndl.zeros_like(param.data)
            
            # Update first moment estimate: u_{t+1} = β₁ u_t + (1-β₁) ∇f(θ_t)
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            
            # Update second moment estimate: v_{t+1} = β₂ v_t + (1-β₂) (∇f(θ_t))²
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            # û_{t+1} = u_{t+1} / (1 - β₁^t)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            # v̂_{t+1} = v_{t+1} / (1 - β₂^t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            # Update parameter: θ_{t+1} = θ_t - α û_{t+1}/(√v̂_{t+1} + ε)
            # Ensure the result maintains the same dtype as the parameter
            update = self.lr * m_hat / (v_hat ** 0.5 + self.eps)
            param.data = ndl.Tensor(param.data.numpy() - update.numpy(), 
                                   dtype=param.dtype, device=param.device)
        ### END YOUR SOLUTION
