import os
import sys
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Features, ClassLabel, Image
import torch

EPOCH = 1
BATCH = 4
LR = 2e-5
WD = 1e-4
args = sys.argv[1:]
if len(args) == 3:
    EPOCH = int(args[0])
    BATCH = int(args[1])
    LR = float(args[2])
dataset_path = "./data/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_images(examples):
    inputs = feature_extractor(examples["image"], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

labels = os.listdir(dataset_path)
labels.sort()
label2id = {v: i for i, v in enumerate(labels)}
id2label = {i: v for v, i in label2id.items()}

features = Features({'image': Image(), 'label': ClassLabel(names=labels)})
dataset = load_dataset('imagefolder', data_dir=dataset_path, features=features, split='train', keep_in_memory=True)

dataset = dataset.train_test_split(test_size=0.15)
encoded_dataset = dataset.map(preprocess_images, batched=True, remove_columns=dataset["train"].column_names)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=len(labels), label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
model = model.to(device)

training_args = TrainingArguments(
    output_dir="./model",
    logging_steps=10,
    logging_dir="./model",
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCH,
    weight_decay=WD,
    save_strategy="no",
    evaluation_strategy="epoch"
)


class ViTDataCollator:
    def __call__(self, batch):
        pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

data_collator = ViTDataCollator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor
)

trainer.train()

model.save_pretrained(f"./model")
torch.save(model, f"./model/model.pt")
