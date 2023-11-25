import torch.nn as nn


class Seq2SeqAutoencoder(nn.Module):
    """
    Модель автоэнкодера. Декодер работает по принципу копирования
    выхода из энкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Seq2SeqAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        return output
