#!/usr/bin/env python3

import torch.nn as nn
import torch as T
import torch.nn.functional as F

from torch.nn.modules.rnn import RNNCellBase
from subLSTM.functional import SubLSTMCell as SubLSTMCellF

import math


class SubLSTMCell(RNNCellBase):
  r"""A long sub-short-term memory (subLSTM) cell, as described in the paper:
  https://arxiv.org/abs/1711.02448

  .. math::

    \begin{array}{ll}
    i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    g = \mathrm{sigmoid}(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
    o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    c' = f * c + g - i \\
    h' = \mathrm{sigmoid}(c') - o \\
    \end{array}

  Args:
    input_size: The number of expected features in the input x
    hidden_size: The number of features in the hidden state h
    bias: If `False`, then the layer does not use bias weights `b_ih` and
      `b_hh`. Default: True

  Inputs: input, (h_0, c_0)
    - **input** (batch, input_size): tensor containing input features
    - **h_0** (batch, hidden_size): tensor containing the initial hidden
      state for each element in the batch.
    - **c_0** (batch. hidden_size): tensor containing the initial cell state
      for each element in the batch.

  Outputs: h_1, c_1
    - **h_1** (batch, hidden_size): tensor containing the next hidden state
      for each element in the batch
    - **c_1** (batch, hidden_size): tensor containing the next cell state
      for each element in the batch

  Attributes:
    weight_ih: the learnable input-hidden weights, of shape
      `(4*hidden_size x input_size)`
    weight_hh: the learnable hidden-hidden weights, of shape
      `(4*hidden_size x hidden_size)`
    bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
    bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

  Examples::

    >>> rnn = nn.SubLSTMCell(10, 20)
    >>> input = Variable(torch.randn(6, 3, 10))
    >>> hx = Variable(torch.randn(3, 20))
    >>> cx = Variable(torch.randn(3, 20))
    >>> output = []
    >>> for i in range(6):
    ...   hx, cx = rnn(input[i], (hx, cx))
    ...   output.append(hx)
  """

  def __init__(self, input_size, hidden_size, bias=True):
    super(SubLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.weight_ih = nn.Parameter(T.Tensor(4 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(T.Tensor(4 * hidden_size, hidden_size))
    if bias:
      self.bias_ih = nn.Parameter(T.Tensor(4 * hidden_size))
      self.bias_hh = nn.Parameter(T.Tensor(4 * hidden_size))
    else:
      self.register_parameter('bias_ih', None)
      self.register_parameter('bias_hh', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input, hx):
    return SubLSTMCellF(
        input, hx,
        self.weight_ih, self.weight_hh,
        self.bias_ih, self.bias_hh,
    )
