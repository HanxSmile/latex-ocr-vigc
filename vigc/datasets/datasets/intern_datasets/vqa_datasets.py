import json
import os

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from io import BytesIO
import random
from .utils import get_det_res_str, compare_imgs, load_det_res


class AOKVQA_SQA_Dataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_path, rationale=False, det_res=None, cot=False):
        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path

        self.rationale = rationale
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.cot = cot

        # read annotation from ceph
        if self.anno_path.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.samples = json.loads(client.get(self.anno_path))
        else:
            self.samples = json.load(open(self.anno_path, 'r'))

        if vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            self.reader = {'type': 'LocalReader', 'body': Image.open}

        self.det_results_dict = None
        if det_res is not None:
            self.det_results_dict = load_det_res(det_res['res_path'])
            compare_imgs([d['image'] for d in self.samples], self.det_results_dict, 'COCOCaptionDataset')

        for idx in range(5):
            data = self.__getitem__(0)
            data.pop('image')
            print('AOKVQADataset: {}'.format(idx), data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ann = self.samples[index]

        question = ann['question']
        choices = ann['choices']
        answer = ann['correct_choice_idx']
        lecture = random.choice(ann['rationales'])
        problem = {
            'question': question,
            'choices': choices,
            'answer': answer,
            'lecture': lecture,
            'solution': '',
            'hint': ''
        }
        mask_text, qa_text, _, _ = self.text_processor(problem)

        img_file = 'train2017/' + str(ann['image_id']).zfill(12) + '.jpg'
        image_path = os.path.join(self.vis_root, img_file)

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
            "mask_text": mask_text,
            "question": question,
            "text_input": qa_text,
            "data_type": "mid_vqa",
            "det_res": det_res_str,
        }
