from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import os
import torch.distributed as dist
import logging
import jsonlines
from vigc.common.dist_utils import main_process
import json


@registry.register_task("instruct_blip_pope_train_val")
class InstructBlipPopeTrainValTask(BaseTask):
    LABEL_PATH = "/mnt/lustre/hanxiao/dpo_work/data/coco_pope_random.json"

    def __init__(self, num_beams, max_len, min_len, use_nucleus_sampling, evaluate, label_path, report_metric=True):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.label_path = label_path

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
        label_path = run_cfg.get("label_path", cls.LABEL_PATH)
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            evaluate=evaluate,
            report_metric=report_metric,
            label_path=label_path
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
            this_sample["id"] = raw_sample["question_id"]
            this_sample["question"] = raw_sample["text"]
            this_sample["answer"] = answer
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

        eval_ret = self.pope_evaluate(eval_result_file, self.label_path)

        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {k: v for k, v in eval_ret.items()}
        # agg_metrics = sum([v for v in eval_ret.values()])
        agg_metrics = sum([v for k, v in eval_ret.items() if k in ("f1",)])
        res["agg_metrics"] = agg_metrics

        return res

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.jsonl" % filename)

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

            result = sorted(result, key=lambda x: int(x["id"]))

            with jsonlines.open(final_result_file, mode="a") as writer:
                for sample in result:
                    writer.write(sample)

            print("result file saved to %s" % final_result_file)

        return final_result_file

    def pope_evaluate(self, ans_file, label_file):

        answers = [json.loads(q) for q in open(ans_file, 'r')]
        label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

        for answer in answers:
            text = answer['answer']

            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]

            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answer['answer'] = 'no'
            else:
                answer['answer'] = 'yes'

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer['answer'] == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))

        result = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": yes_ratio,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN
        }
        return result
