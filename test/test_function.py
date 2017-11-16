#!/usr/bin/env python3

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

from subLSTM.functional import SubLSTMCell as SubLSTMCellF
from subLSTM.nn import SubLSTMCell


def test_function():
  hidden_size = 20
  input_size = 10

  for bias in (True, False):
    weight_ih = T.nn.Parameter(T.Tensor(4 * hidden_size, input_size))
    weight_hh = T.nn.Parameter(T.Tensor(4 * hidden_size, hidden_size))
    bias_ih = T.nn.Parameter(T.Tensor(4 * hidden_size)) if bias else None
    bias_hh = T.nn.Parameter(T.Tensor(4 * hidden_size)) if bias else None

    input = var(T.randn(3, input_size))
    hx = var(T.randn(3, hidden_size))
    cx = var(T.randn(3, hidden_size))
    cell = SubLSTMCellF
    for i in range(6):
      hx, cx = cell(input, (hx, cx), weight_ih, weight_hh, bias_ih, bias_hh)

    hx.sum().backward()

    assert hx.size() == T.Size([3, hidden_size])
    assert cx.size() == T.Size([3, hidden_size])
