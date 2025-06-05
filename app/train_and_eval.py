import random
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Set multiprocessing start method to 'spawn' for compatibility
mp.set_start_method('spawn', force=True)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the ImageFolder-style Dataset
# ─────────────────────────────────────────────────────────────────────────────
pet_dataset = load_dataset(
    "imagefolder",
    data_dir="../PetClassification"
)

# Subset the training dataset to 4,000 images
pet_dataset["train"] = pet_dataset["train"].select(range(4000))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build Label ↔ Index Mappings
# ─────────────────────────────────────────────────────────────────────────────
labels = pet_dataset["train"].features["label"].names
idx2label = {idx: label for idx, label in enumerate(labels)}
label2idx = {label: idx for idx, label in enumerate(labels)}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load Pretrained ViT Model (Base)
# ─────────────────────────────────────────────────────────────────────────────
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    id2label=idx2label,
    label2id=label2idx,
    ignore_mismatched_sizes=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load the ViT Image Processor
# ─────────────────────────────────────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    do_rescale=False,
    return_tensors="pt"
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Define Data Augmentation / Preprocessing Transforms
# ─────────────────────────────────────────────────────────────────────────────
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = v2.Normalize(mean=image_mean, std=image_std)

train_transform = v2.Compose([
    v2.Resize((size, size)),
    v2.RandomHorizontalFlip(0.4),
    v2.RandomVerticalFlip(0.1),
    v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
    v2.RandomApply(transforms=[v2.ColorJitter(brightness=0.3, hue=0.1)], p=0.3),
    v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize,
])

test_transform = v2.Compose([
    v2.Resize((size, size)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize
])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Functions to Apply Transforms to HF Dataset Examples
# ─────────────────────────────────────────────────────────────────────────────
def train_transforms(examples):
    examples["pixel_values"] = [
        train_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples

def test_transforms(examples):
    examples["pixel_values"] = [
        test_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples

# Attach transforms
pet_dataset["train"].set_transform(train_transforms)
pet_dataset["validation"].set_transform(test_transforms)
pet_dataset["test"].set_transform(test_transforms)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric Computation Function
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(preds, labels)}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Collate Function for Batching
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Create DataLoaders for Parallel Processing
# ─────────────────────────────────────────────────────────────────────────────
num_workers = mp.cpu_count()  # Use all 4 cores

train_dataloader = DataLoader(
    pet_dataset["train"],
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

val_dataloader = DataLoader(
    pet_dataset["validation"],
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

test_dataloader = DataLoader(
    pet_dataset["test"],
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    pin_memory=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Configure Training Arguments
# ─────────────────────────────────────────────────────────────────────────────
metric_name = "accuracy"

args = TrainingArguments(
    output_dir="breed-classification",
    evaluation_strategy="steps",
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir="logs",
    remove_unused_columns=False,
    push_to_hub=False,
    hub_model_id=None
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Create Trainer
# ─────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=pet_dataset["train"],
    eval_dataset=pet_dataset["validation"],
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# Override Trainer's internal dataloaders
trainer.train_dataloader = train_dataloader
trainer.eval_dataloader = val_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# 12. Start Training
# ─────────────────────────────────────────────────────────────────────────────
print("\n===== Starting Training =====\n")
trainer.train()
print("\n===== Training Complete =====\n")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Evaluate on Test Set
# ─────────────────────────────────────────────────────────────────────────────
print("Evaluating on the test set...")
test_metrics = trainer.evaluate(pet_dataset["test"], data_loader=test_dataloader)
print(f"\nTest set metrics: {test_metrics}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Save the Fine-Tuned Model & Processor
# ─────────────────────────────────────────────────────────────────────────────
print("Saving the fine-tuned model and processor to 'breed-classification-final' folder...")
output_dir = "../model/breed-classification-final"
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

# ─────────────────────────────────────────────────────────────────────────────
# 15. Demonstrate Inference on a Few Test Images
# ─────────────────────────────────────────────────────────────────────────────
print("\n===== Running Inference on 3 Random Test Images =====\n")
for _ in range(3):
    idx = random.randint(0, len(pet_dataset["test"]) - 1)
    example = pet_dataset["test"][idx]
    pixel_values = test_transform(example["image"].convert("RGB")).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values.to(model.device))
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = idx2label[pred_idx]
    true_label = idx2label[example["label"]]
    print(f"Test Image Index: {idx}")
    print(f"  → True Breed: {true_label}")
    print(f"  → Predicted Breed: {pred_label}\n")

# ─────────────────────────────────────────────────────────────────────────────
# End of script
# ─────────────────────────────────────────────────────────────────────────────