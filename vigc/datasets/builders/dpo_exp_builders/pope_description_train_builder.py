import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dpo_exp_datasets.pope_description_train_dataset import POPEDescriptionTrainDataset_Llava, \
    POPEDescriptionTrainDataset_Minigpt4


@registry.register_builder("pope_description_dpo_train")
class PopeDescriptionDPOTrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = {
        "llava": POPEDescriptionTrainDataset_Llava,
        "minigpt4": POPEDescriptionTrainDataset_Minigpt4,
    }

    DATASET_CONFIG_DICT = {"default": "configs/datasets/dpo_exp/pope_description_dpo_train.yaml"}

    def build_datasets(self):
        logging.info("Building POPE Description DPO Train datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation
        vis_root = build_info.images
        subset = self.config.get("subset", "minigpt4")
        assert subset in ("llava", "minigpt4", "instruct_blip_7b", "instruct_blip_13b")

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls[subset]
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            anno_path=anno_path.get(subset),
            vis_root=vis_root
        )

        _ = datasets[split][0]
        return datasets
