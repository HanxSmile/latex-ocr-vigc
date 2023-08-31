from vigc.datasets.datasets.base_dataset import BaseDataset
import torch


class COCO_Pseudo_Dataset(BaseDataset):

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.samples[index]
        image = self._read_image(ann)

        image = self.vis_processor(image)
        sources = {
            "conversations": [
                {"from": "Human", "value": ann["question"]},
                {"from": "Assistant", "value": ann["answer"]}
            ]
        }
        conv_text = self.text_processor(sources)

        return {
            "image": image,
            "text_input": conv_text,
            'data_type': 'conversation',
            'det_res': "",
        }

    def collater(self, samples):
        image_list, text_input_list, data_type_list, det_res_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            data_type_list.append(sample["data_type"])
            det_res_list.append(sample["det_res"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "data_type": data_type_list,
            "det_res": det_res_list
        }
