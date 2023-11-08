import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.intern_datasets.llava_vigc import LlavaCompDataset, LlavaDescDataset, LlavaConvDataset

TRAIN_DATASET_DICT = {
    "llava_comp": LlavaCompDataset,
    "llava_desc": LlavaDescDataset,
    "llava_conv": LlavaConvDataset,
}

ALL_DATASET_CONFIG_DICT = {
    "llava_comp": "configs/datasets/llava_instruct150k/{task}_intern/trainval_llava_comp.yaml",
    "llava_desc": "configs/datasets/llava_instruct150k/{task}_intern/trainval_llava_desc.yaml",
    "llava_conv": "configs/datasets/llava_instruct150k/{task}_intern/trainval_llava_conv.yaml",
}

class VIGCBuilder(BaseDatasetBuilder):
    TASK = "vig"
    TYPE = None
    DATASET_CONFIG_DICT = {"default": None}

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.

        self.build_processors()

        task = self.TASK
        type_ = self.TYPE
        logging.info(f"Building {type_} {task} Training datasets...")

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = TRAIN_DATASET_DICT[type_]
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            task=task
        )
        _ = datasets['train'][0]

        return datasets


@registry.register_builder("intern_llava_comp_vig")
class LlavaCompVIGBuilder(VIGCBuilder):
    TYPE = "llava_comp"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("intern_llava_desc_vig")
class LlavaDescVIGBuilder(VIGCBuilder):
    TYPE = "llava_desc"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}


@registry.register_builder("intern_llava_conv_vig")
class LlavaConvVIGBuilder(VIGCBuilder):
    TYPE = "llava_conv"
    DATASET_CONFIG_DICT = {"default": ALL_DATASET_CONFIG_DICT[TYPE].format(task="vig")}
