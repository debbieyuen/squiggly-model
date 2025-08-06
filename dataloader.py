# from datasets import load_dataset
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image

# class SquigglyDataset(Dataset):
#     def __init__(self, split="train"):
#         # Load the Hugging Face dataset directly from the imagefolder
#         self.dataset = load_dataset(
#             "imagefolder",
#             repo_id="debbieyuen/squigglydataset",
#             data_dir="data",
#             split=split,
#         )

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = self.transform(item["image"])
#         label = item["label"]  # auto-assigned index
#         return {"images": image, "label": label}






# # dataloader.py — Loads PNG and stroke JSON data

# import os
# import json
# import torch
# from glob import glob
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms

# class SquigglyDataset(Dataset):
#     def __init__(self, data_dir, view='front', max_len=100):
#         self.data_dir = data_dir
#         self.view = view
#         self.max_len = max_len
#         self.samples = []

#         image_files = sorted(glob(os.path.join(data_dir, f"*_{view}.png")))
#         for img_path in image_files:
#             base = os.path.basename(img_path).replace(f"_{view}.png", "")
#             json_path = os.path.join(data_dir, base + ".json")
#             label = base.split("_", 1)[0]
#             if os.path.exists(json_path):
#                 self.samples.append((img_path, json_path, label))

#         self.labels = sorted(list(set(label for _, _, label in self.samples)))
#         self.label_to_idx = {name: i for i, name in enumerate(self.labels)}

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, json_path, label = self.samples[idx]
#         image = Image.open(img_path).convert("RGB")

#         with open(json_path) as f:
#             stroke_data = json.load(f)

#         points = []
#         for stroke in stroke_data:
#             points += [[p["x"], p["y"], p["z"]] for p in stroke["points"]]

#         if len(points) < self.max_len:
#             points += [[0, 0, 0]] * (self.max_len - len(points))
#         else:
#             points = points[:self.max_len]

#         stroke_tensor = torch.tensor(points, dtype=torch.float32)
#         label_idx = self.label_to_idx[label]

#         return image, stroke_tensor, label_idx
