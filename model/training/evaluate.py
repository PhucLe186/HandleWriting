import torch
from inference.decode import decode_predictions
from until.metrics import compute_cer

def evaluate(model, loader, criterion, device, idx2char):
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []

    with torch.no_grad():
        for images, labels, label_lengths, texts in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            T, B, _ = logits.size()
            input_lengths = torch.full((B,), T, dtype=torch.long)

            loss = criterion(logits.log_softmax(2), labels, input_lengths, label_lengths)
            total_loss += loss.item()

            preds = decode_predictions(logits.cpu(), idx2char)

            print(preds[:5])
            print(all_targets[:5])

            all_preds.extend(preds)
            all_targets.extend(texts)

    cer = compute_cer(all_preds, all_targets)
    return total_loss / len(loader), cer
