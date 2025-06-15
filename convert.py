import torch
import torch.nn as nn

class LSTMTextGen(nn.Module):
    def __init__(self, vocab_size=4096, embedding_dim=300, hidden_size=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

class LSTMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, h_0, c_0):
        logits, (h_n, c_n) = self.model(x, (h_0, c_0))
        return logits, h_n, c_n

if __name__ == "__main__":
    model = LSTMTextGen()
    model.load_state_dict(torch.load('spongebob_lstm_3M.pth', weights_only=True))
    model.eval()

    batch_size = 1
    seq_len = 10
    num_layers = 1
    hidden_size = 512

    dummy_input = torch.randint(0, 4096, (batch_size, seq_len), dtype=torch.long)
    h_0 = torch.zeros(num_layers, batch_size, hidden_size)
    c_0 = torch.zeros(num_layers, batch_size, hidden_size)

    wrapper = LSTMWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input, h_0, c_0),
        "spongebob_lstm_with_hidden.onnx",
        export_params=True,
        opset_version=13,
        input_names=['input_ids', 'h_0', 'c_0'],
        output_names=['logits', 'h_n', 'c_n'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'h_0': {1: 'batch_size'},
            'c_0': {1: 'batch_size'},
            'logits': {0: 'batch_size', 1: 'sequence_length'},
            'h_n': {1: 'batch_size'},
            'c_n': {1: 'batch_size'}
        },
        do_constant_folding=True
    )

    print("ONNX model with hidden states saved as spongebob_lstm_with_hidden.onnx")
