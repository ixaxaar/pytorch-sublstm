# subtractive LSTM (subLSTM), for Pytorch

[![Build Status](https://travis-ci.org/ixaxaar/pytorch-sublstm.svg?branch=master)](https://travis-ci.org/ixaxaar/pytorch-sublstm) [![PyPI version](https://badge.fury.io/py/pytorch-sublstm.svg)](https://badge.fury.io/py/pytorch-sublstm)

This is an implementation of subLSTM described in the paper [Cortical microcircuits as gated-recurrent neural networks, Rui Ponte Costa et al.](https://arxiv.org/abs/1711.02448)

## Install

```bash
pip install pytorch-sublstm
```


## Usage

**Parameters**:

Following are the constructor parameters:

| Argument | Default | Description |
| --- | --- | --- |
| input_size | `None` | Size of the input vectors |
| hidden_size | `None` | Size of hidden units |
| num_layers | `1` | Number of layers in the network |
| bias | `True` | Bias |
| batch_first | `False` | Whether data is fed batch first |
| dropout | `0` | Dropout between layers in the network |
| bidirectional | `False` | If the network is bidirectional |


### Example usage:

#### nn Interface
```python
import torch
from torch.autograd import Variable
from subLSTM.nn import SubLSTM

hidden_size = 20
input_size = 10
seq_len = 5
batch_size = 7
hidden = None

input = Variable(torch.randn(batch_size, seq_len, input_size))

rnn = SubLSTM(input_size, hidden_size, num_layers=2, bias=True, batch_first=True)

# forward pass
output, hidden = rnn(input, hidden)
```

#### Cell Interface

```python
import torch
from torch.autograd import Variable
from subLSTM.nn import SubLSTMCell

hidden_size = 20
input_size = 10
seq_len = 5
batch_size = 7
hidden = None

hx = Variable(torch.randn(batch_size, hidden_size))
cx = Variable(torch.randn(batch_size, hidden_size))

input = Variable(torch.randn(batch_size, input_size))

cell = SubLSTMCell(input_size, hidden_size, bias=True)
(hx, cx) = cell(input, (hx, cx))
```

### Attributions:

A lot of the code is recycled from [pytorch](https://pytorch.org)
