from ..base_dataset import BaseDataset
import json
import random
import torch


class POPEEvalDataset(BaseDataset):
    PROMPTS = (
        # "Question: {q} Short answer:",
        "{q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def init_samples(self):
        samples = []
        with open(self.anno_path, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                sample = json.loads(line)
                samples.append(sample)
        return samples

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["text"])

        prompt = random.choice(self.PROMPTS)
        question = prompt.format(q=question)

        input_sample = {
            "image": image,
            "prompt": question
        }

        raw_sample = ann
        return input_sample, raw_sample

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list, candidates = [], [], [], []
        for input_sample, raw_sample in samples:
            raw_sample_list.append(raw_sample)
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])
            candidates.append(["yes", "no"])

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "candidates": candidates,
            "raw_samples": raw_sample_list
        }
