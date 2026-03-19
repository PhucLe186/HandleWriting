import torch
from torchvision import transforms
from PIL import Image
from inference.decode import decode_predictions

def predict(model, image, char2idx, idx2char, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')

    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img)

    pred = decode_predictions(logits.cpu(), idx2char)[0]
    return pred

