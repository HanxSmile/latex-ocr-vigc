from ..base_dataset import BaseDataset
import random
import torch
import json
import os.path as osp


class POPEDescriptionTestDataset(BaseDataset):
    PROMPTS = (
        "{q}",
    )

    QUESTION = "Describe this image in detail."

    def __init__(self, vis_processor, text_processor, vis_root, anno_path):
        with open(osp.join(vis_root, "image_data.json")) as f:
            image_info = json.load(f)
        self.id2path = {int(_["image_id"]): osp.join(*_["url"].split("/")[-2:]) for _ in image_info}
        super(POPEDescriptionTestDataset, self).__init__(vis_processor, text_processor, vis_root, anno_path)

    def init_samples(self):
        # read annotation from ceph
        if self.anno_path.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            samples = json.loads(client.get(self.anno_path))
        else:
            samples = json.load(open(self.anno_path, 'r'))
        for sample in samples:
            sample["image"] = self.id2path[int(sample["image_id"])]
        return samples

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann.get("question", self.QUESTION)).replace("\n", "").replace('\"', '"')

        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        input_sample = {
            "image": image,
            "prompt": question,
        }

        return input_sample, ann

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list = [], [], []
        for input_sample, raw_sample in samples:
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])
            raw_sample_list.append(raw_sample)

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "raw_samples": raw_sample_list
        }
