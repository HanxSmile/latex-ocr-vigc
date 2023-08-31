from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import is_main_process, is_dist_avail_and_initialized
import torch.distributed as dist
import json
import os

VIG_INSTRUCTIONS = {
    "comp":
        "Based on the given image, generate an in-depth reasoning question and then answer it.",
    "conv":
        "Generate a question based on the content of the given image and then answer it.",
    "desc":
        "Generate a question to describe the image content in detail and then answer it."
}


@registry.register_task("intern_vig")
class InternVIGTask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, task, report_metric=False):
        super(InternVIGTask, self).__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate
        self.report_metric = report_metric

        task = task.lower()
        assert task in VIG_INSTRUCTIONS
        self.prompt = VIG_INSTRUCTIONS[task]

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        num_beams = generate_cfg.num_beams
        max_len = generate_cfg.max_len
        min_len = generate_cfg.min_len
        use_nucleus_sampling = generate_cfg.get("use_nucleus_sampling", False)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", False)

        task = run_cfg.llava_task

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
            task=task,
        )

    def valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]
        image = samples["image"]
        bs = int(image.shape[0])
        prompts = [self.prompt] * bs

        inputs = {
            "image": image,
            "prompt": prompts
        }

        qa_pairs = model.caption_generate(
            inputs,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )

        for raw_sample, qa_pair in zip(raw_samples, qa_pairs):
            raw_sample = raw_sample.copy()
            raw_sample["question_answer"] = qa_pair
            results.append(raw_sample)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file, eval_result = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        metrics = {"agg_metrics": 0.0}

        if is_main_process():

            qa_result_file = os.path.join(registry.get_path("result_dir"), "question.json")
            result = []

            for d in eval_result:
                image = d["image"]
                QA = d["question_answer"]
                if "Question:" in QA and "Answer:" in QA:
                    QA = QA.split("Question:")[-1].split("Answer:")
                    if len(QA) == 2:
                        Q, A = QA[0].strip(), QA[1].strip()
                        result.append({"image": image, "question": Q, "answer": A})

            with open(qa_result_file, 'w') as f:
                json.dump(result, f)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return metrics
