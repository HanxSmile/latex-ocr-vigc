from io import BytesIO
from PIL import Image
from ..base_dataset import BaseDataset
import random
import torch
import json
import os.path as osp


class POPEDescriptionTrainDataset_Llava(BaseDataset):
    PROMPTS = (
        "{q}",
    )

    def _read_image(self, sample, image_key="image"):
        image_path = sample[image_key]
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        return image

    def parse_conversation(self, conversations):
        question, answer = None, None
        for sent_info in conversations:
            speaker = sent_info["from"]
            sent = sent_info["value"]
            if speaker == "human" and answer is None:
                question = sent
                continue
            if speaker == "gpt" and question is not None:
                answer = sent
                break
        return question, answer

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))

        chosen_question, chosen = self.parse_conversation(ann["chosen_conversations"])
        reject_question, reject = self.parse_conversation(ann["reject_conversations"])
        question = chosen_question or reject_question

        if question is None or chosen is None or reject is None:
            return self[(index + 1) % len(self)]

        question = self.text_processor(question.replace("<image>", "").replace("\n", "").strip())
        chosen, reject = self.text_processor(chosen.strip()), self.text_processor(reject.strip())

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


class POPEDescriptionTrainDataset_Minigpt4(BaseDataset):
    PROMPTS = (
        "{q}",
    )
    QUESTION = "Describe this image in detail."

    def __init__(self, vis_processor, text_processor, vis_root, anno_path):
        with open(osp.join(vis_root, "image_data.json")) as f:
            image_info = json.load(f)
        self.id2path = {int(_["image_id"]): osp.join(*_["url"].split("/")[-2:]) for _ in image_info}
        super(POPEDescriptionTrainDataset_Minigpt4, self).__init__(vis_processor, text_processor, vis_root, anno_path)

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
        question = self.text_processor(ann.get("question", self.QUESTION))
        chosen, reject = ann["chosen"], ann["reject"]
        if isinstance(chosen, str):
            chosen = [chosen]
        if isinstance(reject, str):
            reject = [reject]
        chosen_score = ann.get("chosen_score", [1] * len(chosen))
        reject_score = ann.get("reject_score", [1] * len(reject))

        chosen = random.choices(chosen, list(chosen_score), k=1)[0]
        reject = random.choices(reject, list(reject_score), k=1)[0]

        chosen = self.text_processor(chosen)
        reject = self.text_processor(reject)

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
