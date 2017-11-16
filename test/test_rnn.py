#!/usr/bin/env pythonbatch_size

import pytest
import numpy as np

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

import sys
import os
import math
import time
sys.path.insert(0, '.')

from subLSTM.nn import SubLSTM


def test_rnn():
  hidden_size = 20
  input_size = 10
  seq_len = 5
  batch_size = 7

  for bias in (True, False):
    for batch_first in (True, False):
      input = var(T.randn(batch_size, seq_len, input_size)) if batch_first else var(T.randn(seq_len, batch_size, input_size))
      hx = None
      rnn = SubLSTM(input_size, hidden_size, num_layers=2, bias=bias, batch_first=batch_first)

      outputs = []
      for i in range(6):
        output, hx = rnn(input, hx)
        outputs.append(output)

      T.stack(outputs).sum().backward()

      assert hx[-1][-1][0].size() == T.Size([batch_size, hidden_size])
      assert hx[-1][-1][1].size() == T.Size([batch_size, hidden_size])



def test_rnn_bidirectional():
  hidden_size = 20
  input_size = 10
  seq_len = 5
  batch_size = 7

  for bias in (True, False):
    for batch_first in (True, False):
      input = var(T.randn(batch_size, seq_len, input_size)) if batch_first else var(T.randn(seq_len, batch_size, input_size))
      hx = None
      rnn = SubLSTM(input_size, hidden_size, num_layers=3, bias=bias, batch_first=batch_first, bidirectional=True)

      outputs = []
      for i in range(6):
        output, hx = rnn(input, hx)
        outputs.append(output)

      T.stack(outputs).sum().backward()

      assert hx[-1][-1][0].size() == T.Size([batch_size, hidden_size])
      assert hx[-1][-1][1].size() == T.Size([batch_size, hidden_size])

