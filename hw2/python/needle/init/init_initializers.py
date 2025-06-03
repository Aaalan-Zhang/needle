import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Xavier uniform initialization: a = gain × sqrt(6 / (fan_in + fan_out))
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Xavier normal initialization: std = gain × sqrt(2 / (fan_in + fan_out))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Kaiming uniform initialization: bound = gain × sqrt(3 / fan_in)
    # Use recommended gain value for ReLU: gain = sqrt(2)
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Kaiming normal initialization: std = gain / sqrt(fan_in)
    # Use recommended gain value for ReLU: gain = sqrt(2)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION