from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from dataset.iam_dataset import IAMDataset
from dataset.collate import collate_fn
from models.CRNN_model import CRNN
from inference.predict import predict
from training.train import trainloop

def main():

    dataset = load_dataset("Teklia/IAM-line")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_texts = [s['text'] for s in dataset['train']]
    vocab = sorted(set(''.join(all_texts)))
    vocab = ['<blank>'] + vocab
    model = CRNN(vocab_size=len(vocab)).to(device)

    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}

    train_dataset = IAMDataset(dataset['train'], char2idx)
    val_dataset   = IAMDataset(dataset['validation'], char2idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    NUM_EPOCHS = 200
    # trainloop(NUM_EPOCHS,model, train_loader, optimizer,val_loader,criterion, device, scheduler, idx2char )

    model.load_state_dict(torch.load('checkpoints/best_crnn.pth'))
    result = predict(model, dataset['test']['image'][1], char2idx, idx2char, device)
    print(f"Prediction: {result}")

    dataset['test']['image'][1].show()

if __name__ == "__main__":
    main()