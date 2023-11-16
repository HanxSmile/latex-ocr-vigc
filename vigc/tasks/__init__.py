"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.tasks.caption_train_eval import InstructBlipCaptionTask
from vigc.tasks.llava_150k_gen import InstructBlipLLavaVIGTask
from vigc.tasks.vqa_train_eval import InstructBlipVQATask
from vigc.tasks.vqg_test import InstructBlipVQGTask
from vigc.tasks.image_text_pretrain import ImageTextPretrainTask
from vigc.tasks.intern_vig import InternVIGTask
from vigc.tasks.llava_vqa_test import LLaVAVQATestTask
from vigc.tasks.dummy_task.hit_word_task import HitWordInferTask
from vigc.tasks.pope_vqa_train_val import InstructBlipPopeTrainValTask
from vigc.tasks.pope_vqa_test import InstructBlipPopeTestTask

from vigc.tasks.pope_description_train_val import InstructBlipDescriptionPopeTrainValTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "InstructBlipCaptionTask",
    "InstructBlipLLavaVIGTask",
    "InstructBlipVQATask",
    "InstructBlipVQGTask",
    "ImageTextPretrainTask",
    "InternVIGTask",
    "LLaVAVQATestTask",
    "HitWordInferTask",
    "InstructBlipPopeTrainValTask",
    "InstructBlipPopeTestTask",
    "InstructBlipDescriptionPopeTrainValTask",
]
