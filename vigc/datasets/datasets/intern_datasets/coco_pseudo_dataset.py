from vigc.datasets.datasets.base_dataset import BaseDataset


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
