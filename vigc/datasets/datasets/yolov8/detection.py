import torch
from vigc.datasets.datasets.base_dataset import BaseDataset
import os.path as osp
from io import BytesIO
from PIL import Image
import json
import logging
from tqdm.auto import tqdm
import numpy as np

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes


class YOLODetectionDataset(BaseDataset):

    def __init__(self, vis_processor, vis_root, anno_path, use_segments=False, use_keypoints=False):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        super().__init__(vis_processor, None, vis_root, anno_path)

    def init_samples(self):
        samples = []
        with open(self.anno_path, "r") as f:
            raw_samples = json.load(f)
        for sample in tqdm(raw_samples, desc="parsing samples..."):
            image_name, label_name = sample["image"], sample["label"]
            image_suffix = image_name.split(".")[-1].lower()
            label_suffix = label_name.split(".")[-1].lower()
            if image_suffix not in IMG_FORMATS or label_suffix != "json":
                logging.warning(f"Invalid image format of '{image_name}' or label format of '{label_name}'")
                continue

            image_path = osp.join(self.vis_root, image_name)
            label_path = osp.join(self.vis_root, label_name)
            if (not osp.isfile(image_path)) or (not osp.isfile(label_path)):
                logging.warning(f"'{image_path}' or '{label_path}' not exist.")
                continue
            label_res = self.read_label(label_path)
            sample_res = {
                "image": image_path,
                "label": label_res
            }
            samples.append(sample_res)

        return samples

    def read_label(self, label_path):
        num_keypoints, keypoints_dim = 0, 0

        with open(label_path) as f:
            label_info = json.load(f)

        classes = np.array(label_info["classes"], dtype=np.float32).reshape(-1, 1)  # (n, 1)
        bboxes = np.array(label_info["bboxes"], dtype=np.float32)  # (n, 4)

        segments, keypoints = None, None
        if self.use_segments:
            segments = self._parse_segments(label_info)
            assert len(classes) == len(segments), \
                f"The number of classes and segments not match in '{label_path}'."

        if self.use_keypoints:
            keypoints, num_keypoints_, keypoints_dim_ = self._parse_keypoints(label_info)
            num_keypoints, keypoints_dim = (num_keypoints or num_keypoints_), (keypoints_dim or keypoints_dim_)
            assert num_keypoints_ == num_keypoints and keypoints_dim_ == keypoints_dim
            assert len(classes) == len(
                keypoints), f"The number of classes and keypoints not match in '{label_path}'."
        assert len(classes) == len(bboxes), f"The number of classes and bboxes not match in '{label_path}'."
        lb = np.concatenate((classes, bboxes), axis=1)
        assert np.min(lb) >= 0
        _, valid_index = np.unique(lb, axis=0, return_index=True)
        if len(valid_index) < len(classes):
            classes = classes[valid_index]
            bboxes = bboxes[valid_index]
            if segments:
                segments = [segments[_] for _ in valid_index]
            if keypoints:
                keypoints = keypoints[valid_index]

        label_res = {
            "classes": classes,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "segments": segments
        }

        return label_res

    def _parse_segments(self, label_info):
        return [np.array(_, dtype=np.float32).reshape(-1, 2) for _ in label_info["segments"]]

    def _parse_keypoints(self, label_info):
        keypoints = np.array(label_info["keypoints"])  # (n, n_k, n_d)
        assert np.max(keypoints) <= 1, f'non-normalized or out of bounds coordinates {keypoints[keypoints > 1]}'
        num_keypoints_, keypoints_dim_ = keypoints.shape[-2:]
        if keypoints_dim_ == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        return keypoints, num_keypoints_, keypoints_dim_

    def __getitem__(self, index):
        ann = self.samples[index]
        try:
            image = self.vis_processor(self._read_image(ann))
        except Exception:
            return self[(index + 1) % len(self)]
        if image is None:
            return self[(index + 1) % len(self)]
        equation = ann["equation"]
        return {"image": image, "text_input": equation, "id": index}

    def _read_image(self, sample, image_key="image"):
        img_file = sample[image_key]
        vis_root = sample["vis_root"]
        image_path = osp.join(vis_root, img_file)
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        return image

    def collater(self, samples):
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "id": id_list
        }
