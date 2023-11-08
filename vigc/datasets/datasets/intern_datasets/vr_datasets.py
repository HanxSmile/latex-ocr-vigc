import json
import os

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from io import BytesIO
import random

from .utils import get_det_res_str, load_det_res, compare_imgs


class VRDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_path, det_res=None):
        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path

        self.vis_processor = vis_processor
        self.text_processor = text_processor

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

        print('total {} vqa samples'.format(self.__len__()))

        self.det_results_dict = None
        if det_res is not None:
            self.det_results_dict = load_det_res(det_res['res_path'])
            compare_imgs([a['image'] for a in self.samples], self.det_results_dict, 'VRDataset')

        # for idx in range(5):
        #     data = self.__getitem__(idx)
        #     data.pop('image')
        #     print('VRDataset: {}'.format(idx), data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ann = self.samples[index]

        question, answer = ann['question'], random.choice(ann['answer'])
        answer, question = answer.strip(), question.strip()

        if 'choice_txt' in ann:
            # choice VQA processor
            choice_txt = ann['choice_txt']
            choice_txt = choice_txt.strip()
            mask_text, qa_text = self.text_processor(question, answer, choice_txt)
        else:
            # normal VQA processor
            mask_text, qa_text = self.text_processor(question, answer)

        img_file = ann['image']
        image_path = os.path.join(self.vis_root, img_file)

        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        image = self.vis_processor(image)

        det_res_str = get_det_res_str(image_path, self.det_results_dict)

        det_res_str = '' if det_res_str is None else det_res_str

        return {
            "image": image,
            'question': question,
            "mask_text": mask_text,
            "text_input": qa_text,
            "data_type": "vqa",
            "det_res": det_res_str,
        }


class ICONQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_path, det_res=None):
        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path

        self.vis_processor = vis_processor
        self.text_processor = text_processor

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

        print('total {} vqa samples'.format(self.__len__()))

        self.det_results_dict = None
        if det_res is not None:
            self.det_results_dict = load_det_res(det_res['res_path'])
            compare_imgs([a['image'] for a in self.samples], self.det_results_dict, 'VRDataset')

        # for idx in range(5):
        #     data = self.__getitem__(idx)
        #     data.pop('image')
        #     print('VRDataset: {}'.format(idx), data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ann = self.samples[index]

        question, answer = ann['question'], random.choice(ann['answer'])
        answer, question = answer.strip(), question.strip()

        if 'choice_txt' in ann:
            # choice VQA processor
            choice_txt = ann['choice_txt']
            choice_txt = choice_txt.strip()
            mask_text, qa_text = self.text_processor(question, answer, choice_txt)
        else:
            # normal VQA processor
            mask_text, qa_text = self.text_processor(question, answer)

        img_file = ann['image']
        image_path = os.path.join(self.vis_root, img_file)

        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        image = self.vis_processor(image)

        det_res_str = get_det_res_str(image_path, self.det_results_dict)

        det_res_str = '' if det_res_str is None else det_res_str

        return {
            "image": image,
            'question': question,
            "mask_text": mask_text,
            "text_input": qa_text,
            "data_type": "mid_vqa",
            "det_res": det_res_str,
        }


GQAVRDataset = VRDataset
VSRVRDataset = VRDataset
