import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dpo_exp_datasets.mme_eval_dataset import MMEEvalDataset


@registry.register_builder("mme_eval")
class MMEEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = MMEEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dpo_exp/mme_eval.yaml"
    }

    def build_datasets(self):
        logging.info("Building MME Eval datasets ...")
        self.build_processors()

        build_info = self.config.build_info
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
        )
        _ = datasets['eval'][0]

        return datasets
