import torch
from torch import nn
import json
import torch.nn.functional as F

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
        print(x)
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

def tokens_to_indices(tokens, vocab):
    # Map tokens to indices; use 0 (<UNK>) if token not in vocab
    return [vocab.get(token, 0) for token in tokens]

def clean_ascii(text):
    return ''.join(c for c in text if ord(c) < 128)

def tokenize(text):
    tokens = []
    current = ""

    for char in text:
        if char.isalpha():
            current += char
        else:
            if current:
                tokens.append(current)
                current = ""
            if char == '\n':
                tokens.append('\n')       # newline is its own token
            elif char.strip() == "":
                tokens.append(' ')        # space or tab
            else:
                tokens.append(char)       # punctuation

    if current:
        tokens.append(current)

    return tokens

def generate(model, input_sequence, vocab, index_to_token, temperature=1.0, hidden=None, max_length=200):
    sentence = ''

    tokens = tokenize(clean_ascii(input_sequence.lower()))

    indices = tokens_to_indices(tokens, vocab)

    for idx in indices[:-1]:
        inp = torch.tensor([[idx]], dtype=torch.long)
        _, hidden = model(inp, hidden)

    inp = torch.tensor([[indices[-1]]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(inp, hidden)  # logits: (1, 1, vocab_size)
            logits = logits[:, -1, :] / temperature
            logits[0, vocab['<UNK>']] = -float('inf') # Make <UNK> impossible to sample
            logits[0, vocab['[']] = -float('inf') # Make [ impossible to sample
            logits[0, vocab[']']] = -float('inf') # Make ] impossible to sample
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1).item()

            next_token = index_to_token[str(next_idx)]

            if next_token == '\n':
                return sentence, hidden

            sentence += next_token

            inp = torch.tensor([[next_idx]], dtype=torch.long)

        return sentence, hidden

def main():
    model = LSTMTextGen()
    model.load_state_dict(torch.load('spongebob_lstm_3M.pth', weights_only=True))
    model.eval()

    l = []

    for i in range(10):
        l.append(i+1)
    

    with torch.no_grad():
        _, (h_n, c_n) = model(torch.tensor([l], dtype=torch.long))
    
    print(c_n.mean())

    with open('vocab.json') as f:
        vocab = json.load(f)

    with open('index_to_token.json') as f:
        index_to_token = json.load(f)

    hidden = None

    while True:
        input_sequence = 'Squidward: '
        input_sequence += input('You: ')
        if input_sequence == 'q' or input_sequence == 'quit':
            exit()
        input_sequence += '\nSpongeBob: '
        sentence, hidden = generate(model, input_sequence, vocab, index_to_token, hidden=hidden)
        print(f'SpongeBob: {sentence}')

if __name__ == '__main__':
    main()
