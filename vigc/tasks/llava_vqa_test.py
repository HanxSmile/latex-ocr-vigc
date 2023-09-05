import os
import json
from vigc.common.registry import registry
from vigc.tasks.base_task import BaseTask
import logging
from vigc.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
import torch.distributed as dist


@registry.register_task("intern_llava_vqa_test")
class LLaVAVQATestTask(BaseTask):
    def __init__(
            self,
            num_beams,
            max_len,
            min_len,
            use_nucleus_sampling,
            file_name,
    ):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.use_nucleus_sampling = use_nucleus_sampling
        self.file_name = file_name

    @classmethod
    def setup_task(cls, cfg):
        num_beams = cfg.run_cfg.generate_cfg.get("num_beams", 5)
        max_len = cfg.run_cfg.generate_cfg.get("max_len", 10)
        min_len = cfg.run_cfg.generate_cfg.get("min_len", 1)
        use_nucleus_sampling = cfg.run_cfg.generate_cfg.get("use_nucleus_sampling", False)
        file_name = cfg.run_cfg.get("file_name", "test_results")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            use_nucleus_sampling=use_nucleus_sampling,
            file_name=file_name,
        )

    def valid_step(self, model, samples):
        answers = model.vqa_generate(
            samples,
            use_nucleus_sampling=self.use_nucleus_sampling,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )

        results = []

        questions = samples["prompt"]
        question_ids = samples["question_id"]
        image_paths = samples["image_path"]

        for qid, question, answer, image_path in zip(question_ids, questions, answers, image_paths):
            results.append({
                "question_id": int(qid),
                "image_path": image_path,
                "prompt": question,
                "answer_id": int(qid),
                "text": answer.replace("Answer:", "").strip(),
                "model_id": "vigc-model",
                "metadata": {},
            })

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename=self.file_name,
            remove_duplicate="question_id",
            sort_key="question_id",
            save_type="jsonl",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate="", sort_key="", save_type="json"):

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

            if sort_key:
                result.sort(key=lambda item: item[sort_key])

            if save_type == "jsonl":
                final_result_file = final_result_file.replace(".json", ".jsonl")
                ans_file = open(final_result_file, "w")
                for data in result:
                    ans_file.write(json.dumps(data) + "\n")
                    ans_file.flush()
                ans_file.close()
            else:
                json.dump(result, open(final_result_file, "w"), indent=2)
            print("result file saved to %s" % final_result_file)

        return final_result_file
