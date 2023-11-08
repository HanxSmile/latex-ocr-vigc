import json
import os

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from io import BytesIO
import random

from .utils import get_det_res_str, load_det_res, compare_imgs

'''
from minigpt4.common.registry import registry ;vis_processor = registry.get_processor_class('blip2_image_eval').from_config()
text_processor = registry.get_processor_class('conversation')(header='')
import webdataset as wds ; from minigpt4.datasets.builders import image_text_pair_builder
data = image_text_pair_builder.GQA_Conv_Builder()
data.vis_processors['train'] = vis_processor
data.text_processors['train'] = text_processor
data = data.build_datasets()['train']
loader = wds.WebLoader(data, num_workers=1, batch_size=1)
batch = next(iter(loader))
'''


class GQA_Conv_Dataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_path, det_res=None):
        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path

        self.vis_processor = vis_processor
        self.text_processor = text_processor

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
        conv_paris = ann['conversations']
        random.shuffle(conv_paris)
        sources = []
        for conv in conv_paris:
            sources.extend(conv)

        conv_text = self.text_processor({'conversations': sources}, short_prompt=True)

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
            "text_input": conv_text,
            'data_type': 'brief_conversation',
            "det_res": det_res_str,
        }
