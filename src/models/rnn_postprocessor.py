import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNPostProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 rnn_type='lstm', dropout_prob=0.5, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2 if bidirectional else 1

        linear_input_size = hidden_size * self.num_directions
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size,
                               hidden_size,
                               num_layers,
                               batch_first=True,
                               dropout=dropout_prob if num_layers > 1 else 0,
                               bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size,
                              hidden_size,
                              num_layers,
                              batch_first=True,
                              dropout=dropout_prob if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        else:
            raise ValueError("Invalid rnn_type. Choose 'lstm' or 'gru'.")

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(linear_input_size, num_classes)

    def forward(self, x, lengths=None):

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.rnn(x)
        
        if lengths is not None:
            rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            rnn_output = packed_output

        rnn_output = self.dropout(rnn_output)

        logits = self.fc(rnn_output)

        return logits 