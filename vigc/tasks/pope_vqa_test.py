from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask


@registry.register_task("instruct_blip_pope_test")
class InstructBlipPopeTestTask(BaseTask):
    YES_PREFIX = "yes"
    NO_PREFIX = "no"

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, report_metric=False):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        num_beams = generate_cfg.num_beams
        max_len = generate_cfg.max_len
        min_len = generate_cfg.min_len
        use_nucleus_sampling = generate_cfg.get("use_nucleus_sampling", False)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []
        raw_samples = samples["raw_samples"]

        samples["prefix"] = [self.YES_PREFIX] * len(samples["prompt"])

        yes_answers = model.generate(
            samples,
            use_nucleus_sampling=self.use_nucleus_sampling,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        samples["prefix"] = [self.NO_PREFIX] * len(samples["prompt"])
        no_answers = model.generate(
            samples,
            use_nucleus_sampling=self.use_nucleus_sampling,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )

        for raw_sample, yes_answer, no_answer in zip(raw_samples, yes_answers, no_answers):
            this_sample = dict()
            raw_sample = raw_sample.copy()
            gt_answer = raw_sample["answer"].strip().lower()
            this_sample["image_id"] = raw_sample["image_id"]
            this_sample["question"] = raw_sample["question"]
            this_sample["answer"] = raw_sample["answer"]

            yes_answer = yes_answer.strip()
            if yes_answer.lower().startswith("answer"):
                yes_answer = yes_answer.replace("answer:", "").replace("Answer:", "")
                yes_answer = yes_answer.replace("answer :", "").replace("Answer :", "")
                yes_answer = yes_answer.strip()

            no_answer = no_answer.strip()
            if no_answer.lower().startswith("answer"):
                no_answer = no_answer.replace("answer:", "").replace("Answer:", "")
                no_answer = no_answer.replace("answer :", "").replace("Answer :", "")
                no_answer = no_answer.strip()

            if gt_answer == "yes":
                chosen_answer = f"{self.YES_PREFIX}{yes_answer}".replace(" ,", ",").replace(" .", ".")
                reject_answer = f"{self.NO_PREFIX}{no_answer}".replace(" ,", ",").replace(" .", ".")
            elif gt_answer == "no":
                chosen_answer = f"{self.NO_PREFIX}{no_answer}".replace(" ,", ",").replace(" .", ".")
                reject_answer = f"{self.YES_PREFIX}{yes_answer}".replace(" ,", ",").replace(" .", ".")
            else:
                raise ValueError(f"The answer must between ('yes' or 'no'), got '{gt_answer}'")
            this_sample["chosen"] = chosen_answer
            this_sample["reject"] = reject_answer

            results.append(this_sample)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        return {"agg_metrics": 0.0}
