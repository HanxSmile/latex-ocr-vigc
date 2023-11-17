from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import torch.distributed as dist
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from vigc.common.dist_utils import main_process
import json

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork",
                   "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


def parse_pred_ans(pred_ans):
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"

    return pred_label


@registry.register_task("instruct_blip_mme_train_val")
class InstructBlipMMETrainValTask(BaseTask):

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
        print(samples["prompt"])
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
            this_sample["id"] = raw_sample["id"]
            this_sample["qid"] = raw_sample["qid"]
            this_sample["question"] = raw_sample["question"]
            this_sample["gt_answer"] = raw_sample["answer"].lower()
            this_sample["pred_answer"] = answer
            results.append(this_sample)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )
        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        eval_ret = self.mme_evaluate(eval_result_file)

        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {k: v for k, v in eval_ret.items()}
        # agg_metrics = sum([v for v in eval_ret.values()])
        agg_metrics = sum([v for k, v in eval_ret.items() if k in ("Perception_scores", "Cognition_scores")])
        res["agg_metrics"] = agg_metrics

        return res

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

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
                result_new = {}
                id_list = []
                for res in result:
                    if res[remove_duplicate] in id_list:
                        continue
                    id_list.append(res[remove_duplicate])
                    question_type, image_name = res["qid"].split("-")

                    gt_ans = res["gt_answer"].lower()
                    pred_ans = parse_pred_ans(res["pred_answer"])
                    question_data = result_new.setdefault(question_type, {})
                    image_data = question_data.setdefault(image_name, [])
                    image_data.append({"question": res["question"], "gt": gt_ans, "pred": pred_ans})
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict

    def mme_evaluate(self, ans_file):
        with open(ans_file) as f:
            eval_results = json.load(f)

        total_score_dict = dict()

        for eval_type, task_name_list in eval_type_dict.items():
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                task_results = eval_results[task_name]
                img_num = len(task_results)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []
                for img_name, img_results in task_results.items():
                    assert len(img_results) == 2
                    img_correct_num = 0
                    for img_item in img_results:
                        question, gt_ans, pred_ans = img_item["question"], img_item["gt"], img_item["pred"]
                        assert gt_ans in ("yes", "no") and pred_ans in ("yes", "no", "other")
                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        img_correct_num += (gt_ans == pred_ans)
                        task_other_ans_num += (pred_ans not in ("yes", "no"))
                    acc_plus_correct_num += (img_correct_num == 2)
                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus

                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v * 100

                task_score_dict[task_name] = task_score
                scores += task_score
            for task_name, task_res_score in task_score_dict.items():
                total_score_dict[f"{eval_type}_{task_name}"] = task_res_score
            total_score_dict[f"{eval_type}_scores"] = scores

        return total_score_dict
