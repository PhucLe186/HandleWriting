import torch
from training.trainer import train_epoch
from training.evaluate import evaluate
import matplotlib.pyplot as plt



def trainloop(NUM_EPOCHS, model, train_loader, optimizer,val_loader,criterion, device, scheduler, idx2char ):
    best_cer = float('inf')
    train_losses = []
    val_losses = []
    val_cers = []
    for epoch in range(NUM_EPOCHS):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_cer = evaluate(model, val_loader, criterion, device, idx2char)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_cers.append(val_cer)

        print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), 'best_crnn.pth')
            print(f"  ✓ Saved best model (CER: {best_cer:.4f})")

    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.show()