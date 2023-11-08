from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import jsonlines
import argparse
import sys
import json
import requests
import os

# output_lock = threading.Lock()

url = "http://api-text-sh.fengkongcloud.com/v2/saas/anti_fraud/text"
url = "http://47.100.12.48/v2/saas/anti_fraud/text"
# access_key = "4Ky6AV4hE0pWLeG1bXNw"
access_key = "JmDn3MlRX10rrqanvge2"

# type = "ALL"
api_type = "TEXTRISK"


def req(d: dict):
    text = d["content"][:9000]
    uid = d["id"]

    payload = {
        "accessKey": access_key,
        "appId": "default",
        "eventId": "output",
        "type": api_type,
        "data": {
            "text": text,
            "tokenId": uid,
        },
    }

    try:
        res = requests.post(url, json=payload)
        result = res.json()
    except Exception as e:
        print(e, file=sys.stderr)
        return None

    if result["code"] != 1100:
        print(result, file=sys.stderr)
        return None

    return result


class HitWordDataset(Dataset):
    def __init__(self, anno_path, dst_path):
        if os.path.isfile(dst_path):
            anno_path = dst_path
        with open(os.path.join(anno_path), 'r') as f:
            self.inner_dataset = f.readlines()

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        data = json.loads(self.inner_dataset[index])

        if "nlp_content_risk_result" in data:
            return data, True

        result = req(data)
        print(result)
        if result is None:
            return data, False

        data["nlp_content_risk_result"] = result
        data['content_risk_provider'] = "数美"

        return data, True

    def collater(self, batch):
        return batch


def run(
        run_id,
        save_dir,
        work_dir='/mnt/petrelfs/share_data/wangbin/mllm/sz_unsafe/xiaohongshu_filter',
        indices=None,
        batch_size=256,
        num_workers=8,
):
    json_list = os.listdir(work_dir)
    json_list = sorted([_ for _ in json_list if _.endswith(".jsonl")])
    json_list = [json_list[_] for _ in indices if _ < len(json_list)]

    assert os.path.isdir(save_dir)
    save_dir = os.path.join(save_dir, "shumei_results")
    os.makedirs(save_dir, exist_ok=True)

    statistic_dic = {
        k: {"total_nums": 0, "success_nums": 0} for k in json_list
    }
    statistic_dst_path = os.path.join(save_dir, f'shumei_statistic_run_{run_id}.txt')
    for file_name in tqdm(json_list, desc="Parsing data"):
        anno_path = os.path.join(work_dir, file_name)
        dst_path = os.path.join(save_dir, file_name.split('.')[0] + '_shumei.jsonl')
        tmp_dst_path = os.path.join(save_dir, file_name.split('.')[0] + '_tmp_shumei.jsonl')

        dataset = HitWordDataset(anno_path, dst_path)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collater,
            drop_last=False
        )

        with jsonlines.open(tmp_dst_path, mode="a") as writer:
            for batch in tqdm(dataloader, desc="parsing"):
                for data, flag in batch:
                    writer.write(data)
                    statistic_dic[file_name]["total_nums"] += 1
                    statistic_dic[file_name]["success_nums"] += flag
        while os.path.isfile(dst_path):
            os.remove(dst_path)
        else:
            os.rename(tmp_dst_path, dst_path)

    text = ""
    for key, item in statistic_dic.items():
        rate = item["success_nums"] / item["total_nums"]
        text += f"indice-{indices.index(json_list.index(key))}-{key}: {rate}\n"

    with open(statistic_dst_path, 'w', encoding="utf-8") as f:
        f.write(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--work-dir", default="/mnt/petrelfs/share_data/wangbin/mllm/sz_unsafe/xiaohongshu_filter")
    parser.add_argument("--seg-index", required=True, type=int)
    parser.add_argument("--indices", type=str, default="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    json_list = os.listdir(args.work_dir)
    json_list = sorted([_ for _ in json_list if _.endswith(".jsonl")])
    total_file_nums = len(json_list)
    all_indices = list(range(total_file_nums))
    seg_nums = 8
    indices = [all_indices[_::seg_nums] for _ in range(seg_nums)]
    if args.indices:
        with open(args.indices) as f:
            indices = json.load(f)
    run(save_dir=args.work_dir, indices=indices[args.seg_index], run_id=args.run_id)
