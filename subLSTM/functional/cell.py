#!/usr/bin/env python3

import torch.nn as nn
import torch as T
import torch.nn.functional as F


def SubLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):


  hx, cx = hidden
  gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

  ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

  ingate = F.sigmoid(ingate)
  forgetgate = F.sigmoid(forgetgate)
  cellgate = F.sigmoid(cellgate)
  outgate = F.sigmoid(outgate)

  cy = (forgetgate * cx) + (cellgate - ingate)
  hy = F.sigmoid(cy) - outgate

  return hy, cy
