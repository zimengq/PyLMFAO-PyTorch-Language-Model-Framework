"""
Define some model utils
https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
"""

import torch
import torch.nn as nn
import torch.nn.init as init


def xavier_uniform_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def weight_init(module):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(module, nn.Conv1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.BatchNorm1d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm3d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight.data)
        init.normal_(module.bias.data)
    elif isinstance(module, nn.LSTM):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.LSTMCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRU):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRUCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
    """
    Draw from a multinomial distribution efficiently
    """
    return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]