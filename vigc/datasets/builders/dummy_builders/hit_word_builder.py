import logging
from vigc.common.registry import registry
from vigc.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vigc.datasets.datasets.dummy_dataset.hit_word_dataset import HitWordDataset


@registry.register_builder("hit_word_data")
class HitWordEvalBuilder(BaseDatasetBuilder):
    eval_dataset_cls = HitWordDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dummy/hit_word.yaml"
    }

    def build_datasets(self):
        logging.info("Building Hit Word eval datasets ...")
        work_dir = self.config.work_dir
        unsafe_words_anno_path = self.config.unsafe_words_anno_path
        indices = self.config.get("indices", None)

        datasets = dict()

        datasets["eval"] = self.eval_dataset_cls(
            work_dir=work_dir,
            unsafe_words_anno_path=unsafe_words_anno_path,
            indices=indices
        )
        _ = datasets["eval"][0]
        return datasets
