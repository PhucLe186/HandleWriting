import torch
import torch.nn as nn
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, label_lengths, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)  # (T, B, vocab)
        T, B, _ = logits.size()
        input_lengths = torch.full((B,), T, dtype=torch.long)

        loss = criterion(logits.log_softmax(2), labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)
