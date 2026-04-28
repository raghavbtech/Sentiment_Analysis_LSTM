import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)

        h0 = torch.zeros(1, x.size(0), 128)
        c0 = torch.zeros(1, x.size(0), 128)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]
        out = self.fc(out)

        return out