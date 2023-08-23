import os
import json
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from .utils import load_det_res, get_det_res_str, compare_imgs


class LLavaInstruct150kDataset(Dataset):
    def __init__(self, vis_processor, text_processor, data_root, det_res=None):

        self.vis_root = os.path.join(data_root, 'train2014')
        self.data_root = data_root

        # read annotation from ceph
        if self.data_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.list_data_dict = json.loads(client.get(os.path.join(data_root, 'llava_instruct_150k.json')))
        else:
            self.list_data_dict = json.load(open(os.path.join(data_root, 'llava_instruct_150k.json'), "r"))

        if self.data_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            self.reader = {'type': 'LocalReader', 'body': Image.open}

        print('total {} conversations'.format(len(self.list_data_dict)))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.det_results_dict = None
        if det_res is not None:
            det_results_dict = load_det_res(det_res['res_path'])
            self.det_results_dict = {k.split('_')[-1]: v for k, v in det_results_dict.items()}
            compare_imgs([d['image'] for d in self.list_data_dict], self.det_results_dict, 'COCOCaptionDataset')

        # for idx in range(5):
        #     data = self.__getitem__(idx)
        #     data.pop('image')
        #     print('LLavaInstruct150kDataset: {}'.format(idx), data)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        sources = self.list_data_dict[index]

        assert 'image' in sources
        if 'image' in sources:
            image_file = 'COCO_train2014_' + sources['image']
            if self.reader['type'] == 'PetrelReader':
                image = Image.open(BytesIO(self.reader['body'](os.path.join(self.vis_root, image_file)))).convert("RGB")
            else:
                image = self.reader['body'](os.path.join(self.vis_root, image_file)).convert("RGB")

            image = self.vis_processor(image)
            conv_text = self.text_processor(sources)

            det_res_str = get_det_res_str(image_file.split('/')[-1], self.det_results_dict)
            det_res_str = '' if det_res_str is None else det_res_str
        else:
            image = None
            conv_text = None
            det_res_str = None

        return {
            "image": image,
            "text_input": conv_text,
            'data_type': 'conversation',
            'det_res': det_res_str,
        }