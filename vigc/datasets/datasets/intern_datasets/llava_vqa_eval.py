import random
from vigc.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate


class LLaVAVQATestDataset(BaseDataset):
    VQGA_INSTRUCTIONS = {
        "conv": "Generate a question based on the content of the given image and then answer it.",
        "detail": "Generate a question to describe the image content in detail and then answer it.",
        "complex": "Based on the given image, generate an in-depth reasoning question and then answer it.",
    }

    VQGA_PROMPTS = (
        " Question: {q}",
    )

    VQA_PROMPTS = (
        "{q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file, with_instruction=True):
        self.with_instruction = with_instruction
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        available = False
        while not available:
            try:
                ann = self.samples[index]

                qid = ann["question_id"]
                image_path = ann["image"]
                image = self.vis_processor(self._read_image(ann))
                question = self.text_processor(ann["question"])

                if self.with_instruction:
                    instruction = self.VQGA_INSTRUCTIONS[ann["question_type"]]
                    prompt = random.choice(self.VQGA_PROMPTS)
                    question = instruction + prompt.format(q=question)
                else:
                    prompt = random.choice(self.VQA_PROMPTS)
                    question = prompt.format(q=question)

                available = True
            except Exception as e:
                print(f"Error while read file idx {index}, {ann}: {e}")
                index = random.randint(0, len(self.annotation) - 1)

        return {
            "question_id": qid,
            "image": image,
            "prompt": question,
            "image_path": image_path
        }
