"""Datasets for loading data."""

from typing import Tuple
import pathlib
import json
import random

import cv2
import torch
from torch.utils import data

from src.train import augmentations as augs


class DetectionDataset(data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_width: int,
        img_height: int,
        validation: bool = False,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())

        _img_exts = [".jpg", ".jpeg", ".png"]
        self.images = []
        for ext in _img_exts:
            self.images += list(data_dir.glob(f"*{ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )
        self.images = {}
        for image in self.meta_data["images"]:
            self.images[image["id"]] = {
                "file_name": data_dir / image["file_name"],
                "annotations": [],
            }

        image_ids = list(self.images.keys())
        random.Random(42).shuffle(image_ids)
        if validation:
            num_val = int(len(self.images) * 0.1)
            keep_ids = image_ids[-num_val:]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }
        else:
            num_val = int(len(self.images) * 0.1)
            keep_ids = image_ids[:-num_val]
            self.images = {
                key: val for key, val in self.images.items() if key in keep_ids
            }

        for anno in self.meta_data["annotations"]:
            if anno["image_id"] in self.images:
                self.images[anno["image_id"]]["annotations"].append(anno)
        self.ids_map = {idx: img_id for idx, img_id in enumerate(self.images.keys())}
        self.len = len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_data = self.images[self.ids_map[idx]]
        image = cv2.imread(str(image_data["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []
        category_ids = []
        for anno in image_data["annotations"]:
            if "bbox" in anno:
                box = torch.Tensor(anno["bbox"])
                box[2:] += box[:2]
                box[0::2].clamp_(0, image.shape[1])
                box[1::2].clamp_(0, image.shape[0])
                if (box[3] - box[1]) * (box[2] - box[0]):
                    boxes.append(box)
                    category_ids.append(anno["category_id"])

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=self.ids_map[idx],
        )

    def __len__(self) -> int:
        return self.len

    def __str__(self) -> str:
        return f"{len(self.images)} images."
