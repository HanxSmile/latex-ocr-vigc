from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import torch.distributed as dist
import logging
import json


@registry.register_task("instruct_blip_description_pope_train_val")
class InstructBlipDescriptionPopeTrainValTask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, report_metric=True):
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

        answers = model.generate(
            samples,
            use_nucleus_sampling=self.use_nucleus_sampling,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        for raw_sample, answer in zip(raw_samples, answers):
            this_sample = dict()
            raw_sample = raw_sample.copy()
            answer = answer.strip()
            if answer.lower().startswith("answer"):
                answer = answer.replace("answer:", "").replace("Answer:", "")
                answer = answer.replace("answer :", "").replace("Answer :", "")
                answer = answer.strip()
            this_sample["image_id"] = raw_sample["image_id"]
            this_sample["response"] = answer
            results.append(this_sample)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            result = {_["image_id"]: _["response"] for _ in result}

            with open(final_result_file, "w") as f:
                json.dump(result, f)

            print("result file saved to %s" % final_result_file)

        return final_result_file
