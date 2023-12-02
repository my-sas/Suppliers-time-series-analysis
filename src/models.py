import math
import torch
import torch.nn as nn


class RepeatAutoencoder(nn.Module):
    """
    Модель автоэнкодера. Декодер принимает на вход последовательность
    из повторённых векторов с выхода энкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RepeatAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1].repeat(x.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        return output


class RepeatAutoencoder1(nn.Module):
    """
    Модель автоэнкодера. Декодер принимает на вход последовательность
    из повторённых векторов с выхода энкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RepeatAutoencoder1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        _, (hidden, _) = self.encoder(X)
        hidden = hidden[-1].repeat(X.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        output = self.fc(output)
        return output


class BidirectionalAutoencoder(nn.Module):
    """
    Модель автоэнкодера. Декодер принимает на вход последовательность
    из повторённых векторов с выхода энкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BidirectionalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        _, (hidden, _) = self.encoder(X)
        hidden = hidden[-1].repeat(X.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        output = self.fc(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.01, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1, 0, 2)


class PositionalAutoencoder(nn.Module):
    """
    Модель автоэнкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(PositionalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        cell = torch.zeros(hidden.shape)
        pos_vectors = self.pos_encoder(torch.zeros((*x.shape[:2], self.hidden_dim)))

        output, _ = self.decoder(pos_vectors, (hidden, cell))
        output = self.fc(output)

        return output
