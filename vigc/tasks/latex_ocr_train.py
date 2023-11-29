from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
from vigc.common.dist_utils import main_process
import os.path as osp
import json
import re
import numpy as np
from torchtext.data import metrics
from Levenshtein import distance


@registry.register_task("latex_ocr_train")
class LatexOCR_Train(BaseTask):

    def __init__(self, temperature, evaluate, report_metric=True, agg_metric="edit_distance"):
        super(LatexOCR_Train, self).__init__()
        self.temperature = temperature
        self.evaluate = evaluate
        self.agg_metric = agg_metric

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        temperature = generate_cfg.get('temperature', .2)
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        agg_metric = run_cfg.get("agg_metric", "edit_distance")

        return cls(
            temperature=temperature,
            evaluate=evaluate,
            report_metric=report_metric,
            agg_metric=agg_metric
        )

    def valid_step(self, model, samples):
        results = []
        image, text = samples["image"], samples["text_input"]
        preds = model.generate(samples, self.temperature)
        pred_tokens = preds["pred_tokens"]
        pred_strs = preds["pred_str"]

        truth_inputs = model.tokenize(text)
        truth_tokens = model.detokenize(truth_inputs["input_ids"], model.tokenizer)
        truth_strs = model.token2str(truth_inputs["input_ids"], model.tokenizer)

        ids = samples["id"]

        for pred_token, pred_str, truth_token, truth_str, id_ in zip(pred_tokens, pred_strs, truth_tokens, truth_strs,
                                                                     ids):
            this_item = {
                "pred_token": pred_token,
                "pred_str": pred_str,
                "truth_str": truth_str,
                "truth_token": truth_token,
                "id": id_
            }
            results.append(this_item)
        return results

    @staticmethod
    def post_process(s: str):
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
        letter = '[a-zA-Z]'
        noletter = '[\W_^\d]'
        names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
            news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
            if news == s:
                break
        return s

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

        edit_dists = []

        with open(eval_result_file) as f:
            results = json.load(f)

        all_pred_tokens = []
        all_truth_tokens = []
        for result in results:
            pred_token, pred_str, truth_token, truth_str = result["pred_token"], result["pred_str"], result[
                "truth_token"], result["truth_str"]
            truth_str, pred_str = self.post_process(truth_str), self.post_process(pred_str)
            if len(truth_str) > 0:
                edit_dists.append(distance(pred_str, truth_str) / len(truth_str))

            all_pred_tokens.append(pred_token)
            all_truth_tokens.append([truth_token])

        bleu_score = metrics.bleu_score(all_pred_tokens, all_truth_tokens)
        edit_distance = np.mean(edit_dists)
        eval_ret = {"bleu": bleu_score, "edit_distance": edit_distance}

        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                osp.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in eval_ret.items()}
        # agg_metrics = sum([v for v in eval_ret.values()])
        if self.agg_metric == "edit_distance":
            agg_metrics = (1 - edit_distance) * 100
        else:  # bleu
            agg_metrics = bleu_score * 100

        coco_res["agg_metrics"] = agg_metrics

        return coco_res
