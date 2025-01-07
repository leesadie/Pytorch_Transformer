import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformer import Transformer

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# special tokens
src_pad_idx = tokenizer.pad_token_id
trg_pad_idx = tokenizer.pad_token_id
src_vocab_size = tokenizer.vocab_size
trg_vocab_size = tokenizer.vocab_size

# example data
src_sentences = [
    "hello how are you",
    "what is your name",
    "nice to meet you",
    "have a great day"
]
trg_sentences = [
    "bonjour comment ça va",
    "quel est ton nom",
    "ravi de te rencontrer",
    "passe une bonne journée"
]

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, tokenizer, max_length):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.src_sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        trg = self.tokenizer(
            self.trg_sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        return src, trg
    
max_length = 20
dataset = TranslationDataset(src_sentences, trg_sentences, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
device = "cpu"

# transformer model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=512,
    num_layers=6,
    forward_expansion=4,
    heads=8,
    dropout=0.1,
    device=device,
    max_length=max_length
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            # forward pass
            output = model(src, trg[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

            # backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

train_model(model, dataloader, optimizer, criterion, num_epochs=10, device=device)

# inference with translation
def translate_sentence(model, sentence, tokenizer, src_pad_idx, trg_pad_idx, max_length, device):
    model.eval()
    tokens = tokenizer(
        sentence,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"].to(device)

    outputs = [trg_pad_idx]  # start with <pad> or <sos> token
    for _ in range(max_length):
        trg_tensor = torch.tensor([outputs]).to(device)
        with torch.no_grad():
            output = model(tokens, trg_tensor)
        next_token = output.argmax(2)[:, -1].item()
        outputs.append(next_token)
        if next_token == trg_pad_idx:  # end at <eos>
            break

    return tokenizer.decode(outputs, skip_special_tokens=True)

# test sentence
test_sentence = "hello how are you"
translation = translate_sentence(model, test_sentence, tokenizer, src_pad_idx, trg_pad_idx, max_length, device)
print(f"Translated: {translation}")