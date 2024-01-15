from vigc.datasets.datasets.base_dataset import BaseDataset
import os.path as osp
import numpy as np
from .utils import exif_size, segments2boxes
import logging
from tqdm.auto import tqdm
import os
from vigc.common.dist_utils import is_main_process


class YoloDetection(BaseDataset):
    IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes

    def __init__(
            self,
            vis_processor,
            media_root,
            anno_path,
            task,
            kpt_shape=(0, 0),
            class_names=(),
            include_classes=None,
            single_class=False,
            cache_labels=True
    ):
        super().__init__(vis_processor, None, media_root, anno_path)
        self.use_segments = task in ("segment", "segmentation")
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.class_names = class_names
        self.num_cls = len(self.class_names)
        self.cache_labels = cache_labels
        self.cache_path = osp.join(media_root, "labels.cache")
        self.include_classes = include_classes
        self.single_class = single_class

        if self.use_keypoints:  # confirm the shape of keypoints
            assert kpt_shape is not None
            self.nkpt, self.ndim = kpt_shape
            assert self.nkpt > 0 and self.ndim in (2, 3)

        self.labels = self.get_labels()
        self.update_labels()

    def __len__(self):
        return len(self.labels)

    def _verify_image(self, im_file):
        im = self._read_image({"image": im_file})
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # height , width
        assert shape[0] > 9 and shape[1] > 9, f"image size {shape} <10 pixels"
        assert im.format.lower() in self.IMG_FORMATS, f"invalid image format {im.format}"
        return im, shape

    def _verify_label(self, lb_file):
        lb_file = osp.join(self.vis_root, lb_file)
        # Number of (missing, found, empty, corrupt), message, segments, keypoints
        nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None

        if not osp.isfile(lb_file):
            nm = 1  # label missing
            lb = np.zeros((0, (5 + self.nkpt * 3) if self.use_keypoints else 5), dtype=np.float32)
            keypoints = lb[:, 5:].reshape(-1, self.nkpt, 3)
            lb = lb[:, :5]
            return lb, segments, keypoints, nm, nf, ne, nc, msg

        nf = 1  # label found
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]

        nl = len(lb)

        if nl == 0:
            ne = 1  # label empty
            lb = np.zeros((0, 5), dtype=np.float32)
            if self.use_keypoints:
                keypoints = np.zeros(0, self.nkpt, 3)
            return lb, segments, keypoints, nm, nf, ne, nc, msg

        if any(len(_) > 6 for _ in lb) and (not self.use_keypoints):  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
        lb = np.array(lb, dtype=np.float32)

        if self.use_keypoints:
            assert lb.shape[1] == (
                    5 + self.nkpt * self.ndim), f"labels require {(5 + self.nkpt * self.ndim)} columns each"
            points = lb[:, 5:].reshape(-1, self.ndim)[:, :2]
        else:
            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
            points = lb[:, 1:]
        assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
        assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
        # All labels
        assert lb[:, 0].max() < self.num_cls, (
            f"Label class {int(lb[:, 0].max())} exceeds dataset class count {self.num_cls}. "
            f"Possible class labels are 0-{self.num_cls - 1}"
        )

        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            lb = lb[i]  # remove duplicates
            if segments:
                segments = [segments[x] for x in i]
            msg = f"WARNING ⚠️ {lb_file}: {nl - len(i)} duplicate labels removed"

        if self.use_keypoints:
            keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
            if self.ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return lb, segments, keypoints, nm, nf, ne, nc, msg

    @staticmethod
    def load_dataset_cache_file(path):
        """Load an Ultralytics *.cache dictionary from path."""
        import gc

        gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
        cache = np.load(str(path), allow_pickle=True).item()  # load dict
        gc.enable()
        return cache

    @staticmethod
    def save_dataset_cache_file(path, x):
        """Save an Ultralytics dataset *.cache dictionary x to path."""
        if not is_main_process():
            return
        np.save(str(path), x)  # save cache for next time
        if path.endswith(".npy"):
            os.rename(path, path.replace(".npy", ""))
        logging.info(f"New cache created: {path}")

    def get_labels(self):
        if osp.isfile(self.cache_path):
            return self.load_dataset_cache_file(self.cache_path)
        all_labels = []
        for sample in tqdm(self.samples, desc="parsing labels..."):
            try:
                image, shape = self._verify_image(sample["image"])
                lb, segments, keypoints, nm, nf, ne, nc, msg = self._verify_label(sample["annotation"])
            except Exception as e:
                msg = f"WARNING ⚠️ {sample['image']}: ignoring corrupt image/label: {e}"
                logging.warning(msg)
                continue
            if msg:
                logging.warning(msg)

            this_sample = {
                "image": sample["image"],
                "shape": shape,
                "cls": lb[:, :1],
                "bboxes": lb[:, 1:],
                "segments": segments,
                "keypoints": keypoints,
                "normalized": True,
                "bbox_format": "xywh"
            }
            all_labels.append(this_sample)
        if self.cache_labels:
            self.save_dataset_cache_file(self.cache_path, all_labels)
        return all_labels

    def update_labels(self):
        include_class_array = np.array(self.include_classes).reshape(1, -1)
        for i in range(len(self.labels)):
            if self.include_classes is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_class:
                self.labels[i]["cls"][:, 0] = 0
