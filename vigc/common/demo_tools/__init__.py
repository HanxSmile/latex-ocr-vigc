from vigc.models import load_model_and_preprocess
import torch
import argparse

MODEL_TYPE = "vicuna7b"
MODEL_NAME = "blip2_vicuna_instruct"

MODEL_CKPT = {
    "minigpt4": {
        "finetuned": "/mnt/petrelfs/share_data/wangbin/mllm/vigc/models/minigpt4_7b_d3e4.pth",
        "pretrained": "/mnt/petrelfs/hanxiao/input/instruct-blip/blip2_pretrained_flant5xxl.pth"},
    "instruct_blip": {
        "finetuned": "/mnt/petrelfs/share_data/wangbin/mllm/vigc/models/bak_0805/instruct-blip_7b_best.pth",
        "pretrained": "/mnt/petrelfs/hanxiao/input/instruct-blip/instruct_blip_vicuna7b_trimmed.pth"},
}

VIGA_INSTRUCTIONS = {
    "complex reasoning":
        "Based on the given image, generate an in-depth reasoning question and then answer it.",
    "conversation":
        "Generate a question based on the content of the given image and then answer it.",
    "detail description":
        "Generate a question to describe the image content in detail and then answer it."
}


def _update(
        conversation,
        text,
        step,
        answer_length,
        in_section=True,
        last_infer_all=True,
        gen_style="vqga"
):
    last_flag = step == answer_length
    if conversation["question"] is None:  # update question and current text
        questions = []
        ori_answers = []
        for i, QA in enumerate(text):
            Q = None
            A = None
            if "Question:" in QA and "Answer:" in QA:
                QA = QA.split("Question:")[-1].split("Answer:")
                if len(QA) == 2:
                    Q = QA[0].strip()
                    A = QA[1].strip()
            questions.append(Q)
            ori_answers.append(A)
            if Q is None:
                conversation["valid"][i] = False
        conversation["question"] = questions
        conversation["original_answers"] = ori_answers

        current_texts = []

        for i, (c, q) in enumerate(zip(conversation["instruction"], conversation["question"])):
            current_text = c
            if q:
                current_text = f"{current_text} Question: {q} Answer:" if gen_style == "vqga" else q
            current_texts.append(current_text)
        conversation["current_text"] = current_texts
    elif not in_section:
        current_answers = []
        if conversation["corrected_answers"] is None:
            conversation["corrected_answers"] = text
        for answer in text:
            A = ""
            if "." in answer:
                A = answer.split(".")[0].strip() + "."
            if last_flag and last_infer_all:
                A = answer
            current_answers.append(A)
        current_texts = []
        answers = []
        for i, (c, old_a, a) in enumerate(
                zip(conversation["current_text"], conversation["answer"], current_answers)):
            current_text = f"{c} {a}".strip()
            current_texts.append(current_text)
            answer = f"{old_a} {a}".strip()
            answers.append(answer)
        conversation["current_text"] = current_texts
        conversation["answer"] = answers
    else:  # in_section
        current_answers = []
        first_flag = True
        if conversation["corrected_answers"] is None:
            conversation["corrected_answers"] = text
        else:
            first_flag = False
        for i, answer in enumerate(text):
            A = answer.split("\n\n")[0].strip()
            if last_flag and last_infer_all:
                A = answer.strip()
            current_answers.append(A)
        current_texts = []
        answers = []
        for i, (c, old_a, a) in enumerate(
                zip(conversation["current_text"], conversation["answer"], current_answers)):
            current_text = f"{c} {a}".strip() if first_flag else f"{c} \n\n{a}".strip()
            current_texts.append(current_text)
            conversation["answer_lst"][i].append(a)
            answer = f"{old_a} \n\n{a}".strip()
            answers.append(answer)
        conversation["current_text"] = current_texts
        conversation["answer"] = answers

    return conversation


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--device1", default="0")
    parser.add_argument("--device2", default="0")
    parser.add_argument("--ckpt_minigpt4", default="")
    parser.add_argument("--ckpt_instruct_blip", default="")
    args = parser.parse_args()
    return args


def prepare_models(args):
    device1 = torch.device(f"cuda:{args.device1}") if torch.cuda.is_available() else "cpu"
    device2 = torch.device(f"cuda:{args.device2}") if torch.cuda.is_available() else "cpu"

    print('Loading minigpt4 model...')
    minigpt4_model, minigpt4_processors, _ = load_model_and_preprocess(
        name=MODEL_NAME,
        model_type=MODEL_TYPE,
        is_eval=True,
        device=device1
    )

    minigpt4_model.load_checkpoint(MODEL_CKPT["minigpt4"]["pretrained"])
    minigpt4_model.load_checkpoint(args.ckpt_minigpt4 or MODEL_CKPT["minigpt4"]["finetuned"])

    print('Loading intruct blip...')

    instruct_blip_model, instruct_blip_vis_processors, _ = load_model_and_preprocess(
        name=MODEL_NAME,
        model_type=MODEL_TYPE,
        is_eval=True,
        device=device2,
    )

    instruct_blip_model.load_checkpoint(MODEL_CKPT["instruct_blip"]["pretrained"])
    instruct_blip_model.load_checkpoint(args.ckpt_instruct_blip or MODEL_CKPT["instruct_blip"]["finetuned"])

    print('Loading model done!')
    res = {
        "device1": device1,
        "device2": device2,
        "minigpt4_model": minigpt4_model,
        "instruct_blip_model": instruct_blip_model,
        "minigpt4_processors": minigpt4_processors,
        "instruct_blip_vis_processors": instruct_blip_vis_processors
    }
    return res


def inference(all_elements):
    device1 = all_elements["device1"]
    device2 = all_elements["device2"]
    minigpt4_model = all_elements["minigpt4_model"]
    instruct_blip_model = all_elements["instruct_blip_model"]
    minigpt4_processors = all_elements["minigpt4_processors"]
    instruct_blip_vis_processors = all_elements["instruct_blip_vis_processors"]

    def _inner_f(
            image,
            question: str,
            task,
            min_len,
            max_len,
            beam_size,
            temperature,
            answer_length,
            in_section,
            model_type,
    ):
        use_minigpt4 = model_type.lower() == "minigpt4"
        last_infer_all = True
        answer_length = int(answer_length)
        prompt = VIGA_INSTRUCTIONS[task]
        in_section = in_section == "In Paragraph"

        if use_minigpt4:
            model = minigpt4_model
            vis_processors = minigpt4_processors
            device = device1
        else:
            model = instruct_blip_model
            vis_processors = instruct_blip_vis_processors
            device = device2

        print(image, question, task, min_len, max_len, beam_size, model_type, temperature)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        instructions = [prompt]
        question = question.strip().capitalize()
        if question:
            question = [question]
            answer_length -= 1
            current_text = question
        else:
            question = None
            current_text = instructions

        all_res = {
            "instruction": instructions,
            "current_text": current_text,
            "answer": [""] * len(instructions),
            "answer_lst": [list() for _ in instructions],
            "question": question,
            "original_answers": None,
            "corrected_answers": None,
            "valid": [True] * len(instructions)
        }

        for i in range(answer_length + 1):
            this_sample = {"prompt": all_res["current_text"], "image": image}
            answers = model.generate(
                this_sample,
                num_beams=beam_size,
                max_length=max_len,
                min_length=min_len,
                temperature=temperature,
            )
            _update(all_res, answers, step=i, answer_length=answer_length, in_section=in_section,
                    last_infer_all=last_infer_all)

        question = all_res["question"][0]
        answer = all_res["answer"][0]
        res = f"Question: {question} \n\nAnswer: {answer}"
        return res

    return _inner_f
