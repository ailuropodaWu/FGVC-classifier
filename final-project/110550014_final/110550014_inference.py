import os
import sys
import csv
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from tqdm import tqdm

model_path = './110550014_model.pt'         # path to the pretrained model
data_path = './test'                        # path to the test data
output_csv = './110550014_prediction.csv'   # path to output csv file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
model = model.to(device)
model.eval()  

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]

with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])  

    for filename in tqdm(os.listdir(data_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(data_path, filename)
            predicted_class = predict_image(image_path)

            writer.writerow([os.path.splitext(filename)[0], predicted_class])

print(f"Predictions written to {output_csv}")
