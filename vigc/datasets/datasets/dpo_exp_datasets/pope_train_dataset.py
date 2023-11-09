from ..base_dataset import BaseDataset
import random
import torch
import json
import os.path as osp


class POPETrainDataset(BaseDataset):
    PROMPTS = (
        "{q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_path):
        with open(osp.join(vis_root, "image_data.json")) as f:
            image_info = json.load(f)
        self.id2path = {int(_["image_id"]): osp.join(_["url"].split("/")[-2:]) for _ in image_info}
        super(POPETrainDataset, self).__init__(vis_processor, text_processor, vis_root, anno_path)

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
        question = self.text_processor(ann["question"])
        chosen = self.text_processor(ann["chosen"])
        reject = self.text_processor(ann["reject"])

        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        input_sample = {
            "image": image,
            "question": question,
            "chosen": chosen,
            "reject": reject
        }

        return input_sample

    def collater(self, samples):
        image_list, question_list, chosen_list, reject_list = [], [], [], []
        for input_sample in samples:
            image_list.append(input_sample["image"])
            question_list.append(input_sample["question"])
            chosen_list.append(input_sample["chosen"])
            reject_list.append(input_sample["reject"])

        return {
            "image": torch.stack(image_list, dim=0),
            "question": question_list,
            "chosen": chosen_list,
            "reject": reject_list
        }
