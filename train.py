# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import ViTModel
from dataloader import SquigglyDataset

# Define ViT-based classifier
class SquigglyViTClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, images):
        b, v, c, h, w = images.size()
        images = images.view(b * v, c, h, w)

        outputs = self.vit(pixel_values=images)
        cls_embeddings = outputs.last_hidden_state[:, 0]  # CLS token

        cls_embeddings = cls_embeddings.view(b, v, -1).mean(dim=1)
        logits = self.classifier(self.dropout(cls_embeddings))
        return logits

def main():
    dataset = SquigglyDataset()
    label_set = sorted(set(example["label"] for example in dataset))
    label_map = {label: idx for idx, label in enumerate(label_set)}

    def collate_fn(batch):
        images = torch.stack([item["images"] for item in batch])
        strokes = [item["strokes"] for item in batch]  # Optional
        labels = torch.tensor([label_map[item["label"]] for item in batch])
        return images, strokes, labels

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SquigglyViTClassifier(num_labels=len(label_set)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for images, _, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    main()