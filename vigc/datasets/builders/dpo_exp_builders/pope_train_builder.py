import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dpo_exp_datasets.pope_train_dataset import POPETrainDataset


@registry.register_builder("pope_dpo_train")
class PopeDPOTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = POPETrainDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/dpo_exp/pope_dpo_train.yaml"}

    def build_datasets(self):
        logging.info("Building POPE DPO Train datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            anno_path=anno_path,
            vis_root=vis_root
        )

        _ = datasets[split][0]
        return datasets
