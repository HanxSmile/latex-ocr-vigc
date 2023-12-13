import torch
from .base_dataset import BaseDataset
import os.path as osp
import glob
from io import BytesIO
from PIL import Image


class Im2LatexDataset(BaseDataset):

    def init_samples(self):
        samples = []
        for vis_root, anno_path in zip(self.vis_root, self.anno_path):
            images = [path.replace('\\', '/') for path in glob.glob(osp.join(vis_root, '*.png'))]
            indices = [int(osp.basename(img).split('.')[0]) for img in images]

            eqs = open(anno_path, 'r').read().split('\n')
            eqs = [eqs[_] for _ in indices]

            for i, e in zip(images, eqs):
                samples.append({"image": i, "equation": e, "vis_root": vis_root})
        return samples

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
