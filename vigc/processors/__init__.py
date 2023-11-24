"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from vigc.processors.base_processor import BaseProcessor

from vigc.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)

from vigc.processors.vqa_processors import (
    ScienceQATextProcessor,
    ConversationTextProcessor,
    VQATextProcessor
)
from vigc.processors.formula_processor import (
    FormulaImageTrainProcessor,
    FormulaImageEvalProcessor
)

from vigc.common.registry import registry

__all__ = [
    "BaseProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    # Intern-v0
    "ScienceQATextProcessor",
    "ConversationTextProcessor",
    "VQATextProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
