import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNPostProcessor(nn.Module):
    """RNN model (LSTM or GRU) for post-processing frame-level probabilities."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 rnn_type='lstm', dropout_prob=0.5, bidirectional=True):
        """
        Args:
            input_size (int): Dimension of input features per frame (e.g., 3 * NUM_CLASSES).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes (NUM_CLASSES + 1 for background).
            rnn_type (str): Type of RNN ('lstm' or 'gru').
            dropout_prob (float): Dropout probability.
            bidirectional (bool): If True, use a bidirectional RNN.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2 if bidirectional else 1

        # Input dimension for the linear layer depends on bidirectionality
        linear_input_size = hidden_size * self.num_directions

        # Define the RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size,
                               hidden_size,
                               num_layers,
                               batch_first=True,
                               dropout=dropout_prob if num_layers > 1 else 0, # Dropout only between layers
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

        # Define the dropout layer (applied after RNN)
        self.dropout = nn.Dropout(dropout_prob)

        # Define the output linear layer
        self.fc = nn.Linear(linear_input_size, num_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass through the RNN.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_size).
            lengths (Tensor, optional): Tensor of sequence lengths for each batch item, 
                                         used for packing. Shape (batch,).
        
        Returns:
            Tensor: Output logits of shape (batch, seq_len, num_classes).
        """
        
        # Initialize hidden state (and cell state for LSTM)
        # batch_size = x.size(0)
        # h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device) if self.rnn_type == 'lstm' else None

        # Packing sequences (optional but recommended if using batches with padding)
        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)

        # RNN forward pass
        if self.rnn_type == 'lstm':
            # packed_output, (hn, cn) = self.rnn(x, (h0, c0))
            packed_output, _ = self.rnn(x) # Let LSTM initialize hidden state by default
        else: # GRU
            # packed_output, hn = self.rnn(x, h0)
            packed_output, _ = self.rnn(x) # Let GRU initialize hidden state by default
            
        # Unpacking sequences
        if lengths is not None:
            rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            rnn_output = packed_output # If no packing was done

        # Apply dropout
        rnn_output = self.dropout(rnn_output)

        # Apply linear layer to each time step
        logits = self.fc(rnn_output)

        return logits 