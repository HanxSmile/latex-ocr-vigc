import os
import os.path as osp

from ..base_dataset import BaseDataset
import random
import torch


class MMEEvalDataset(BaseDataset):
    PROMPTS = (
        "{q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root):
        super().__init__(vis_processor, text_processor, vis_root, "")

    def init_samples(self):
        dataset_path = self.vis_root
        samples = []
        for folder in os.listdir(dataset_path):
            data_path = os.path.join(dataset_path, folder)
            if not (osp.isdir(data_path) and folder != "eval_tool"):
                continue
            vis_root = osp.join(data_path, "images")
            anno_path = osp.join(data_path, "question_answers_YN")
            if not (osp.isdir(vis_root) and osp.isdir(anno_path)):
                continue
            ann_files = [_ for _ in os.listdir(anno_path) if _.endswith(".txt")]
            ann_names = [_.split(".")[0] for _ in ann_files]
            image_files = [_ for _ in os.listdir(vis_root) if _.endswith(".png") or _.endswith(".jpg")]
            image_names = [_.split(".")[0] for _ in image_files]
            ann_file_dic = {k: v for k, v in zip(ann_names, ann_files)}
            image_file_dic = {k: v for k, v in zip(image_names, image_files)}

            valid_name = set(ann_file_dic.keys()) | set(image_file_dic.keys())
            for name in valid_name:
                image_file = image_file_dic[name]
                ann_file = ann_file_dic[name]
                image_path = osp.join(folder, "images", image_file)
                ann_path = osp.join(anno_path, ann_file)
                with open(ann_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        ann = line.split("\t")
                        question, answer = ann[0].strip(), ann[1].strip()
                        samples.append(
                            {"id": len(samples), "image": image_path, "question_type": folder, "question": question,
                             "answer": answer, "image_path": osp.join(vis_root, image_file)})
        return samples

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])

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
