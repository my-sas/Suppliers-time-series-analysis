import math
import torch
import torch.nn as nn


class RepeatAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RepeatAutoencoder1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_):
        _, (hidden, _) = self.encoder(input_)
        hidden = hidden[-1].repeat(input_.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        output = self.fc(output)
        return output


class GatedMergeUnit(nn.Module):
    def __init__(self, input_dim):
        super(GatedMergeUnit, self).__init__()

        self.fc_gate = nn.Linear(input_dim * 2, input_dim)
        self.fc_transform = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)

        x_gate = torch.sigmoid(self.fc_gate(x))
        x_transform = torch.tanh(self.fc_transform(x))

        return x_gate * x_transform + (1 - x_gate) * x1


class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(BidirectionalEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder_forward = nn.LSTM(input_dim, output_dim, num_layers, batch_first=True)
        self.encoder_backward = nn.LSTM(input_dim, output_dim, num_layers, batch_first=True)

        self.gate = GatedMergeUnit(output_dim)

    def forward(self, input_):
        _, (hidden_forward, _) = self.encoder_forward(input_)
        _, (hidden_backward, _) = self.encoder_backward(input_.flip(dims=[1]))

        output = self.gate(hidden_forward, hidden_backward)
        return output


class BidirectionalDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(BidirectionalDecoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.decoder_forward = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)
        self.decoder_backward = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)

        self.fc_forward = nn.Linear(input_dim, output_dim)
        self.fc_backward = nn.Linear(input_dim, output_dim)

    def forward(self, input_, seq_len):
        input_ = input_[-1].repeat(seq_len, 1, 1).permute(1, 0, 2)

        hidden_forward, _ = self.decoder_forward(input_)
        hidden_backward, _ = self.decoder_backward(input_)

        output_forward = self.fc_forward(hidden_forward)
        output_backward = self.fc_backward(hidden_backward)

        output = torch.stack((output_forward, output_backward))
        return output


class BidirectionalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BidirectionalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = BidirectionalEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = BidirectionalDecoder(hidden_dim, output_dim, num_layers)

    def forward(self, input_):
        hidden = self.encoder(input_)
        output = self.decoder(hidden, input_.shape[1])
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
