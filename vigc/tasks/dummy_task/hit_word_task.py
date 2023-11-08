from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import os
import torch.distributed as dist
import logging
import json
from tqdm.auto import tqdm


@registry.register_task("hit_word_infer_task")
class HitWordInferTask(BaseTask):
    def __init__(self, evaluate, report_metric=True):
        super().__init__()
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        report_metric = run_cfg.get("report_metric", True)
        evaluate = run_cfg.evaluate

        return cls(
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        return samples

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=""
        )

        metrics = {"agg_metrics": 0.0}

        return metrics

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            all_data = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                all_data += res

            statistic_dic = {}
            hit_res = {}

            for data in tqdm(all_data):
                high_hit_word = data["high_hit_word"]
                if high_hit_word is None:
                    continue
                raw_data = data["raw_data"]
                raw_data["high_hit_word"] = high_hit_word
                file_name = data["file_name"]
                if file_name not in statistic_dic:
                    statistic_dic[file_name] = {}
                    hit_res[file_name] = []
                if high_hit_word not in statistic_dic[file_name]:
                    statistic_dic[file_name][high_hit_word] = 0
                statistic_dic[file_name][high_hit_word] += 1
                hit_res[file_name].append(raw_data)

            for file_name, res in hit_res.items():
                save_path = os.path.join(
                    result_dir, file_name.split('.')[0] + '_hits.json'
                )

                statistic_save_path = os.path.join(
                    result_dir, file_name.split('.')[0] + '_hits_statistic.json'
                )
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False)

                with open(statistic_save_path, "w", encoding="utf-8") as f:
                    json.dump(statistic_dic[file_name], f, ensure_ascii=False)

            print("result file saved to %s" % result_dir)

        return result_dir
