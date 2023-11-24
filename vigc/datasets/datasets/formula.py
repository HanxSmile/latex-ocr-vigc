import torch
from .base_dataset import BaseDataset
import os.path as osp
import glob


class Im2LatexDataset(BaseDataset):

    def init_samples(self):
        images = [path.replace('\\', '/') for path in glob.glob(osp.join(self.vis_root, '*.png'))]
        indices = [int(osp.basename(img).split('.')[0]) for img in images]

        eqs = open(self.anno_path, 'r').read().split('\n')
        eqs = [eqs[_] for _ in indices]
        samples = []
        for i, e in zip(images, eqs):
            samples.append({"image": i, "equation": e})
        return samples

    def __getitem__(self, index):
        ann = self.samples[index]
        image = self.vis_processor(self._read_image(ann))
        if image is None:
            return self[(index + 1) % len(self)]
        equation = ann["equation"]
        return {"image": image, "text_input": equation, "id": index}

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
