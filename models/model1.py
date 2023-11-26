import torch.nn as nn


class Seq2SeqAutoencoder(nn.Module):
    """
    Модель автоэнкодера. Декодер работает по принципу копирования
    выхода из энкодера.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Seq2SeqAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(hidden)
        return output
