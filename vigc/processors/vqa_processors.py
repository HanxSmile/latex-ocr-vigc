"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import random

from vigc.common.registry import registry
from vigc.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from string import Template


def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem):
    txt_context = problem['hint']
    context = " ".join([txt_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt, options[probelm['answer']]


def get_rotate_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    options = options[:len(choices)]
    random.shuffle(options)

    for i, c in enumerate(options):
        choice_list.append("{}. {}".format(options[i], choices[i]))
    choice_list.sort()
    choice_txt = " ".join(choice_list)
    return choice_txt, options[probelm['answer']]


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    lecture = problem['lecture']
    return lecture


def get_solution_text(problem):
    solution = problem['solution']
    return solution


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True,
                       split_token='###'):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    mask_text = input + " " + split_token + ' Assistant:'
    mask_text = mask_text.replace("  ", " ").strip()
    text = input + " " + split_token + ' Assistant: ' + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return mask_text, text, split_token


@registry.register_processor("science_qa")
class ScienceQATextProcessor(BaseProcessor):
    def __init__(self, prompt="", input_format='QCM-ALE', max_words=50, rotate=False):
        self.prompt = prompt
        self.input_format = input_format
        self.max_words = max_words
        self.rotate = rotate
        self.options = ["A", "B", "C", "D", "E"]

    def __call__(self, problem):
        question = get_question_text(problem)
        context = get_context_text(problem)
        if self.rotate:
            choice, answer = get_rotate_choice_text(problem, self.options)

        else:
            choice, answer = get_choice_text(problem, self.options)
        lecture = get_lecture_text(problem)
        solution = get_solution_text(problem)

        mask_text, qa_text, question_split = create_one_example(self.input_format,
                                                                question,
                                                                context,
                                                                choice,
                                                                answer,
                                                                lecture,
                                                                solution,
                                                                test_example=False, split_token='###')

        mask_text = self.prompt + mask_text
        qa_text = self.prompt + qa_text

        return mask_text, qa_text, question_split, question

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        input_format = cfg.get("input_format", "")
        max_words = cfg.get("max_words", 50)
        rotate = cfg.get("rotate", "")

        return cls(prompt=prompt, input_format=input_format, max_words=max_words, rotate=rotate)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


short_prompts = [
    'answer the following question briefly, ',
    'answer briefly, ',
    'answer this question briefly, ',
    'answer this question with a few words, ',
    'give a short answer to this question, ',
]


def _add_speaker_and_signal(header, source, get_conversation=True, short_prompt=False):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = 'Human'
            if short_prompt:
                ex_prompt = random.choice(short_prompts)
            else:
                ex_prompt = ''
            temp = (BEGIN_SIGNAL + from_str + ": " + ex_prompt +
                    sentence["value"] + END_SIGNAL)
        else:
            from_str = 'Assistant'
            temp = (BEGIN_SIGNAL + from_str + ": " +
                    sentence["value"] + END_SIGNAL)

        if get_conversation:
            conversation += temp

    return conversation


'''
"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questio
ns.\n\n### Human: <Img>ImageHere<Img> What are the colors of the bus in the image?\n### Assistant: The bus in the image is white and red.\n### Human: What feature ca
n be seen on the back of the bus?\n### Assistant: The back of the bus features an advertisement.\n### Human: Is the bus driving down the street or pulled 
off to the side?\n### Assistant: The bus is driving down the street, which is crowded with people and other vehicles.\n### "
'''


@registry.register_processor("conversation")
class ConversationTextProcessor(BaseProcessor):
    def __init__(self, header, prompt=""):
        self.header = f"{header}\n\n"
        self.prompt = prompt

    @classmethod
    def from_config(cls, cfg=None):
        return cls(cfg.header)

    def __call__(self, source, short_prompt=False):
        # print("--------------conversation----------------")
        conversation = _add_speaker_and_signal(self.header, source['conversations'], short_prompt=short_prompt)
        # print(conversation)
        del_header_conv = conversation.replace("<image>", "").split("### Human:", 1)[-1].lstrip()
        return del_header_conv


@registry.register_processor("vqa")
class VQATextProcessor(BaseProcessor):
    def __init__(self):
        self.prompt_template = [
            "$question",
            "Question: $question Answer:",
            "$question A short answer to the question is",
            "Q: $question A:",
            "Question: $question Short answer:",
            "Given the image, answer the following question. $question",
            "Based on the image, respond to this question with a short answer: $question. Answer:",
            "Use the provided image to answer the question: $question Provide your answer as short as possible:",
            "What is the answer to the following question? $question",
            "The question '$question' can be answered using the image. A short answer is"
        ]
        self.prompt_template = [Template(s) for s in self.prompt_template]

        """
        <prompt>
        QUESTION: ....
        CHOICES: ....
        ANSWER: ...
        """
        self.choice_prompt_template = [
            "Given the image, answer the following question.\n",
            "Based on the image, respond to this question with a short answer.\n",
            "Use the provided image to answer the question.\n",
            "What is the answer to the following question?\n"
        ]

    def __call__(self, question, answer, choice_txt=None):

        # good sentence ends with '.'
        if answer[-1] != '.':
            answer += '.'

        if choice_txt is not None:
            choice_txt = choice_txt.strip()
            prompt = random.choice(self.choice_prompt_template)
            question = "QUESTION: " + question + '\n'
            choice_txt = "CHOICES: " + choice_txt + '\n'
            mask_text = prompt + question + choice_txt + '### Assistant: '
            qa_text = mask_text + 'ANSWER: ' + answer

            # mask_text = prompt + question + choice_txt + '### Assistant: ' + 'ANSWER: '
            # qa_text = mask_text + answer
        else:
            question = random.choice(self.prompt_template).substitute(question=question)
            mask_text = question + '\n' + '### Assistant: '
            qa_text = mask_text + answer

        return mask_text, qa_text
