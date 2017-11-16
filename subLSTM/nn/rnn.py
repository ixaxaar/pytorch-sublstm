#!/usr/bin/env python3

import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
from torch.nn import Module

from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence as pack, pad_packed_sequence as pad

from subLSTM.functional import SubLSTMCell as SubLSTMCellF

import math


class SubLSTM(Module):

  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers=1,
      bias=True,
      batch_first=False,
      dropout=0,
      bidirectional=False
  ):
    super(SubLSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.dropout_state = {}
    self.bidirectional = bidirectional
    num_directions = 2 if bidirectional else 1

    gate_size = 4 * hidden_size

    self._all_weights = []
    for layer in range(num_layers):
      for direction in range(num_directions):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions

        w_ih = nn.Parameter(T.Tensor(gate_size, layer_input_size))
        w_hh = nn.Parameter(T.Tensor(gate_size, hidden_size))
        b_ih = nn.Parameter(T.Tensor(gate_size))
        b_hh = nn.Parameter(T.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)

        suffix = '_reverse' if direction == 1 else ''
        param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
        if bias:
          param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
        param_names = [x.format(layer, suffix) for x in param_names]

        for name, param in zip(param_names, layer_params):
          setattr(self, name, param)
        self._all_weights.append(param_names)

    self.flatten_parameters()
    self.reset_parameters()

  def flatten_parameters(self):
    pass

  def _apply(self, fn):
    ret = super(SubLSTM, self)._apply(fn)
    self.flatten_parameters()
    return ret

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input, hx=None):
    timesteps = input.size(1) if self.batch_first else input.size(0)
    directions = 2 if self.bidirectional else 1
    is_packed = isinstance(input, PackedSequence)

    if is_packed:
      input, batch_sizes = pad(input)
      max_batch_size = batch_sizes[0]
    else:
      batch_sizes = None
      max_batch_size = input.size(0) if self.batch_first else input.size(1)

    # layer * direction
    if hx is None:
      num_directions = 2 if self.bidirectional else 1
      hx = var(input.data.new(max_batch_size, self.hidden_size).zero_(), requires_grad=False)
      hx = (hx, hx)
      hx = [[hx for x in range(directions)] for d in range(self.num_layers)]

    # make weights indexable with layer -> direction
    ws = self.all_weights
    if directions == 1:
      ws = [ [w] for w in ws ]
    else:
      ws = [ [ws[l*2], ws[l*2+1]] for l in range(self.num_layers) ]

    # make input batch-first, separate into timeslice wise chunks
    input = input if self.batch_first else input.transpose(0, 1)
    os = [[input[:, i, :] for i in range(timesteps)] for d in range(directions)]
    if directions > 1:
      os[1].reverse()

    for time in range(timesteps):
      for layer in range(self.num_layers):
        for direction in range(directions):

          if self.bias:
            (w_ih, w_hh, b_ih, b_hh) = ws[layer][direction]
          else:
            (w_ih, w_hh) = ws[layer][direction]
            b_ih = None
            b_hh = None

          hy, cy = SubLSTMCellF(os[direction][time], hx[layer][direction], w_ih, w_hh, b_ih, b_hh)
          hx[layer][direction] = (hy, cy)
          os[direction][time] = hy

        if directions > 1:
          os[0][time] = T.cat([ os[d][time] for d in range(directions) ], -1)
          os[1][time] = os[0][time]

    output = T.stack([T.stack(o, 1) for o in os])
    output = T.cat(output, -1) if self.bidirectional else output[0]
    output = output if self.batch_first else output.transpose(0, 1)

    if is_packed:
      output = pack(output, batch_sizes)
    return output, hx

  def __repr__(self):
    s = '{name}({input_size}, {hidden_size}'
    if self.num_layers != 1:
      s += ', num_layers={num_layers}'
    if self.bias is not True:
      s += ', bias={bias}'
    if self.batch_first is not False:
      s += ', batch_first={batch_first}'
    if self.dropout != 0:
      s += ', dropout={dropout}'
    if self.bidirectional is not False:
      s += ', bidirectional={bidirectional}'
    s += ')'
    return s.format(name=self.__class__.__name__, **self.__dict__)

  def __setstate__(self, d):
    super(SubLSTM, self).__setstate__(d)
    self.__dict__.setdefault('_data_ptrs', [])
    if 'all_weights' in d:
      self._all_weights = d['all_weights']
    if isinstance(self._all_weights[0][0], str):
      return
    num_layers = self.num_layers
    num_directions = 2 if self.bidirectional else 1
    self._all_weights = []
    for layer in range(num_layers):
      for direction in range(num_directions):
        suffix = '_reverse' if direction == 1 else ''
        weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
        weights = [x.format(layer, suffix) for x in weights]
        if self.bias:
          self._all_weights += [weights]
        else:
          self._all_weights += [weights[:2]]

  @property
  def all_weights(self):
    return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
