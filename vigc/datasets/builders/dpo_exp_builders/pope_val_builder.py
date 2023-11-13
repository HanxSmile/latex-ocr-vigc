import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dpo_exp_datasets.pope_val_dataset import POPEEvalDataset


@registry.register_builder("pope_vqa_eval")
class POPEVQAEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = POPEEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dpo_exp/pope_coco_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building POPE COCO Random Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation
        vis_root = build_info.images

        subset = self.config.get("subset", "random")
        assert subset in ("random", "popular", "adversarial")

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_file=anno_path.get(subset),
        )
        _ = datasets['eval'][0]

        return datasets
