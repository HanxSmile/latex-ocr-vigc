import os
import logging
import warnings

from vigc.common.registry import registry
from .base_dataset_builder import BaseDatasetBuilder

from vigc.datasets.datasets.intern_datasets.cc_sbu_dataset import CCSBUAlignDataset
from vigc.datasets.datasets.intern_datasets.vqa_datasets import AOKVQA_SQA_Dataset
from vigc.datasets.datasets.intern_datasets.science_qa_dataset import ScienceQADataset
from vigc.datasets.datasets.intern_datasets.llava_instruct150k_dataset import LLavaInstruct150kDataset
from vigc.datasets.datasets.intern_datasets.gqa_conversation_datasets import GQA_Conv_Dataset
from vigc.datasets.datasets.intern_datasets.vr_datasets import VSRVRDataset
from vigc.datasets.datasets.intern_datasets.coco_pseudo_dataset import COCO_Pseudo_Dataset


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info(f"Building datasets: cc_sbu_align...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]

        return datasets


@registry.register_builder("aokvqa_sqa")
class AOKVQA_SQA_Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQA_SQA_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/a-okvqa/sqa_format.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets: aokvqa...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path = build_info.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            rationale=True,
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]

        return datasets


@registry.register_builder("science_qa")
class ScienceQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ScienceQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/science_qa/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets: science_qa...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_root=storage_path,
            phase='train',
            filter_pure_text=self.config.filter_pure_text,
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]
        return datasets


@registry.register_builder("llava_instruct150k")
class LLavaInstruct150kBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLavaInstruct150kDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava_instruct150k/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets: llava_instruct150k...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_root=storage_path,
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]

        return datasets


@registry.register_builder("vqav2_conv")
class VQAv2_Conv_Builder(BaseDatasetBuilder):
    train_dataset_cls = GQA_Conv_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/defaults_conv.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info(f"Building datasets: gqa_conversation...")
        self.build_processors()

        build_info = self.config.build_info
        ann_paths = build_info.annotations,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=ann_paths,
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]

        return datasets


@registry.register_builder("vsr_vr")
class GQAVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRVRDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsr/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info(f"Building datasets: vsr_vr...")
        self.build_processors()

        build_info = self.config.build_info
        ann_paths = build_info.annotations,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=ann_paths,
            det_res=build_info.get('det_res', None),
        )
        datasets['train'][0]

        return datasets


@registry.register_builder("pseudo_coco")
class InternPseudoCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCO_Pseudo_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_pseudo/intern_default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info(f"Building COCO Pseudo for Intern...")
        self.build_processors()

        build_info = self.config.build_info
        ann_path = self.config.annotation,
        vis_root = build_info.images

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=ann_path,
        )
        datasets['train'][0]

        return datasets
