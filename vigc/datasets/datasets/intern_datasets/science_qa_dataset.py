import json
import os
from io import BytesIO
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_det_res_str, load_det_res, compare_imgs

'''
from minigpt4.common.registry import registry ;vis_processor = registry.get_processor_class('blip2_image_eval').from_config()
text_processor = registry.get_processor_class('science_qa').from_config()
import webdataset as wds ; from minigpt4.datasets.builders import image_text_pair_builder
data = image_text_pair_builder.GQA_Conv_Builder()
data.vis_processors['train'] = vis_processor
data.text_processors['train'] = text_processor
data = data.build_datasets()['train']
loader = wds.WebLoader(data, num_workers=1, batch_size=1)
batch = next(iter(loader))
'''


class ScienceQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, data_root, phase, filter_pure_text=True, det_res=None):
        self.vis_root = os.path.join(data_root, phase)
        pure_text = not filter_pure_text
        self.pure_text = pure_text

        problems_paths = os.path.join(data_root, 'problems.json')
        pid_splits = os.path.join(data_root, 'pid_splits.json')

        # read annotation from ceph
        if self.vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            problems = json.loads(client.get(problems_paths))
            pid_splits = json.loads(client.get(pid_splits))
        else:
            problems = json.load(open(problems_paths))
            pid_splits = json.load(open(pid_splits))

        # read images from ceph
        if self.vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            self.reader = {'type': 'LocalReader', 'body': Image.open}

        qids = pid_splits[phase]
        if self.pure_text:
            self.problems = [(qid, problems[qid]) for qid in qids if problems[qid]['image'] is None]
            print('total {} pure text problems'.format(len(self.problems)))
        else:
            self.problems = [(qid, problems[qid]) for qid in qids if problems[qid]['image'] is not None]
            print('total {} image-text problems'.format(len(self.problems)))
            self.det_results_dict = None
            if det_res is not None:
                self.det_results_dict = load_det_res(det_res['res_path'])
                compare_imgs([a[1]['image'] for a in self.problems], self.det_results_dict, 'ScienceQADataset')

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.problems)

    def __get_vl_item__(self, index):
        # TODO this assumes image input, not general enough
        qid, problem = self.problems[index]
        mask_text, qa_text, question_split, question = self.text_processor(problem)

        image_path = os.path.join(self.vis_root, qid, problem["image"])
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        image = self.vis_processor(image)

        det_res_str = get_det_res_str(image_path.split('/')[-1], self.det_results_dict)

        det_res_str = '' if det_res_str is None else det_res_str

        return {
            "image": image,
            "question": question,
            "mask_text": mask_text,
            "text_input": qa_text,
            "question_split": question_split,
            "data_type": "long_vqa",
            "det_res": det_res_str,
        }

    def __get_pure_text_item__(self, index):
        qid, problem = self.problems[index]
        mask_text, qa_text, question_split, question = self.text_processor(problem)
        return {
            "image": torch.zeros((3, 224, 224)),
            "question": question,
            "mask_text": mask_text,
            "text_input": qa_text,
            "question_split": question_split,
            "data_type": "pure_conversation",
            "det_res": '',
        }

    def __getitem__(self, index):
        if self.pure_text:
            return self.__get_pure_text_item__(index)
        else:
            return self.__get_vl_item__(index)
