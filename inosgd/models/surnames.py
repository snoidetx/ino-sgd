import torch
from torch import nn
from opacus.layers import DPLSTM


VOCAB_SIZE = 259


class SurnamesLSTM(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        output_size,
        num_lstm_layers=1,
        bidirectional=False,
        vocab_size=VOCAB_SIZE,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = DPLSTM(
            embedding_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden=None):
        x = self.embedding(x)  # -> [B, T, D]
        x, _ = self.lstm(x, hidden)  # -> [B, T, H]
        x = x[:, -1, :]  # -> [B, H]
        x = self.out_layer(x)  # -> [B, C]
        return x
    