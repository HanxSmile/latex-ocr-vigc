"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.datasets.builders.base_dataset_builder import load_dataset_config

from vigc.common.registry import registry

from vigc.datasets.builders.caption_builder import CCSBUBuilder
from vigc.datasets.builders.vic_eval_builder import (
    AOKVQAEvalBuilder,
    VQAv2EvalBuilder,
    OKVQAEvalBuilder,
    LLaVAVQATestBuilder
)

from vigc.datasets.builders.vig_builder import (
    LlavaCompVIGBuilder,
    LlavaDescVIGBuilder,
    LlavaConvVIGBuilder,
    A_OKVQA_VIGBuilder,
    OKVQA_VIGBuilder,
    VQAv2_VIGBuilder,
    COCO_Pseudo_VIGBuilder,
    LlavaCompVQABuilder,
    LlavaDescVQABuilder,
    LlavaConvVQABuilder,
    A_OKVQA_VQABuilder,
    OKVQA_VQABuilder,
    VQAv2_VQABuilder,
    COCO_Pseudo_VQABuilder,
    LlavaCompVQGBuilder,
    LlavaDescVQGBuilder,
    LlavaConvVQGBuilder,
    A_OKVQA_VQGBuilder,
    OKVQA_VQGBuilder,
    COCO_Pseudo_VQGBuilder,
)
from vigc.datasets.builders.vig_eval_builder import (
    AOKVQAEvalBuilder,
    COCO_Jiahui_VQGBuilder,
    COCOPseudoEvalBuilder,
    OKVQAEvalBuilder,
    VQAv2EvalBuilder,
    LlavaVQGAEvalBuilder,
)

from vigc.datasets.builders.intern_builder import (
    CCSBUAlignBuilder,
    AOKVQA_SQA_Builder,
    ScienceQABuilder,
    LLavaInstruct150kBuilder,
    VQAv2_Conv_Builder,
    GQAVRBuilder,
    InternPseudoCOCOBuilder
)

from vigc.datasets.builders.intern_vigc_builder import *
from vigc.datasets.builders.dummy_builders.hit_word_builder import HitWordEvalBuilder
from vigc.datasets.builders.dpo_exp_builders.pope_val_builder import POPEVQAEvalBuilder
from vigc.datasets.builders.dpo_exp_builders.pope_train_builder import PopeDPOTrainBuilder
from vigc.datasets.builders.dpo_exp_builders.pope_test_builder import POPEVQATestBuilder

from vigc.datasets.builders.dpo_exp_builders.pope_description_train_builder import PopeDescriptionDPOTrainBuilder
from vigc.datasets.builders.dpo_exp_builders.pope_description_test_builder import POPEDescriptionTestBuilder
from vigc.datasets.builders.dpo_exp_builders.mme_eval_builder import MMEEvalBuilder
from vigc.datasets.builders.formula import FormulaRecTrainBuilder, FormulaRecEvalBuilder, \
    MultiScaleFormulaRecTrainBuilder

__all__ = [
    # "AOKVQA_Train_Builder",
    "VQAv2EvalBuilder",
    "OKVQAEvalBuilder",
    "AOKVQAEvalBuilder",
    "LlavaCompVIGBuilder",
    "LlavaDescVIGBuilder",
    "LlavaConvVIGBuilder",
    "A_OKVQA_VIGBuilder",
    "OKVQA_VIGBuilder",
    "VQAv2_VIGBuilder",
    "COCO_Pseudo_VIGBuilder",
    "LlavaCompVQABuilder",
    "LlavaDescVQABuilder",
    "LlavaConvVQABuilder",
    "A_OKVQA_VQABuilder",
    "OKVQA_VQABuilder",
    "VQAv2_VQABuilder",
    "COCO_Pseudo_VQABuilder",
    "LlavaCompVQGBuilder",
    "LlavaDescVQGBuilder",
    "LlavaConvVQGBuilder",
    "A_OKVQA_VQGBuilder",
    "OKVQA_VQGBuilder",
    "COCO_Pseudo_VQGBuilder",
    "AOKVQAEvalBuilder",
    "COCO_Jiahui_VQGBuilder",
    "COCOPseudoEvalBuilder",
    "OKVQAEvalBuilder",
    "VQAv2EvalBuilder",
    "LlavaVQGAEvalBuilder",
    "CCSBUBuilder",

    "CCSBUAlignBuilder",
    "AOKVQA_SQA_Builder",
    "ScienceQABuilder",
    "LLavaInstruct150kBuilder",
    "VQAv2_Conv_Builder",
    "GQAVRBuilder",
    "InternPseudoCOCOBuilder",
    "LLaVAVQATestBuilder",

    "POPEVQAEvalBuilder",
    "PopeDPOTrainBuilder",
    "POPEVQATestBuilder",

    "PopeDescriptionDPOTrainBuilder",
    "POPEDescriptionTestBuilder",

    "MMEEvalBuilder",

    "FormulaRecTrainBuilder",
    "FormulaRecEvalBuilder",
    "MultiScaleFormulaRecTrainBuilder",
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
                data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
