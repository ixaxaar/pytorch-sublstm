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


def test_cell():
  hidden_size = 20
  input_size = 10

  for bias in (True, False):
    input = var(T.randn(3, input_size))
    hx = var(T.randn(3, hidden_size))
    cx = var(T.randn(3, hidden_size))

    cell = SubLSTMCell(input_size, hidden_size, bias=bias)

    for i in range(6):
      (hx, cx) = cell(input, (hx, cx))

    hx.sum().backward()
    assert hx.size() == T.Size([3, hidden_size])
    assert cx.size() == T.Size([3, hidden_size])
