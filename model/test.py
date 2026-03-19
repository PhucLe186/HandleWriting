from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)


image = Image.open("C:/Users/HP/Downloads/test3.jpg").convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values)

text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(text)
