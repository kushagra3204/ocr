import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.root = root
        self.transforms = transforms
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_map = {cat['name']: cat['id'] for cat in self.categories}
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(f"{self.root}/{path}").convert("RGB")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.tensor([w * h])
        target["iscrowd"] = torch.tensor([0])

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.img_ids)