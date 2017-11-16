# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, LSTM or subLSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40        # Train a subLSTM on PTB with CUDA, reaching perplexity of 117.61
python generate.py                      # Generate samples from the trained subLSTM model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU`, `nn.LSTM` and `sublstm.SubLSTM`).
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, subLSTM)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```
