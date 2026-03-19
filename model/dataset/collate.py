import torch

def collate_fn(batch):
    images, labels, texts = zip(*batch)

    images = torch.stack(images, 0)  # (B, 1, H, W)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths, texts