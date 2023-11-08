import os
import json
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import jsonlines
import argparse


class HitWordDataset(Dataset):
    def __init__(self, anno_path,
                 unsafe_words_anno_path='/mnt/petrelfs/share_data/ouyanglinke/clean_data/common/high.jsonl'):
        with open(unsafe_words_anno_path, 'r') as f:
            hit_words = f.readlines()

        unsafe_words = []
        for hit in hit_words:
            hit = json.loads(hit)
            unsafe_words.append(hit['word'])
        self.unsafe_words = unsafe_words
        self.compile_unsafe_words = [re.compile(_) for _ in self.unsafe_words]

        with open(os.path.join(anno_path), 'r') as f:
            self.inner_dataset = f.readlines()

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        data = json.loads(self.inner_dataset[index])
        result = None
        for hit_word, compile_unsafe_word in zip(self.unsafe_words, self.compile_unsafe_words):
            result_search = re.search(compile_unsafe_word, data["content"])
            # print(result_search)
            if result_search:
                data['high_hit_word'] = hit_word
                result = data
                break
        return result

    def collater(self, batch):
        return batch


def run(
        save_dir,
        work_dir='/mnt/petrelfs/share_data/wangbin/mllm/sz_unsafe/xiaohongshu/',
        indices=None,
        batch_size=256,
        num_workers=8,
):
    json_list = os.listdir(work_dir)
    json_list = sorted([_ for _ in json_list if _.endswith(".jsonl")])
    json_list = [json_list[_] for _ in indices if _ < len(json_list)]

    for file_name in tqdm(json_list, desc="Parsing data"):
        anno_path = os.path.join(work_dir, file_name)
        dst_path = os.path.join(save_dir, file_name.split('.')[0] + '_hits.jsonl')
        dst_statistic_path = os.path.join(save_dir, file_name.split('.')[0] + '_hits_statistic.txt')
        if os.path.isfile(dst_path):
            os.remove(dst_path)
        if os.path.isfile(dst_statistic_path):
            os.remove(dst_statistic_path)
        dataset = HitWordDataset(anno_path)
        statistic_info = {k: 0 for k in dataset.unsafe_words}
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collater,
            drop_last=False
        )

        with jsonlines.open(dst_path, mode="a") as writer:
            for batch in tqdm(dataloader, desc="parsing"):
                for data in batch:
                    if data is None:
                        continue
                    writer.write(data)
                    statistic_info[data["high_hit_word"]] += 1

        text = ""
        for key, item in statistic_info.items():
            text += f"{key}: {item}\n"

        with open(dst_statistic_path, 'w', encoding="utf-8") as f:
            f.write(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--seg-index", required=True, type=int)
    parser.add_argument("--save-dir", required=True, type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    work_dir = '/mnt/petrelfs/share_data/wangbin/mllm/sz_unsafe/xiaohongshu/'
    json_list = os.listdir(work_dir)
    json_list = sorted([_ for _ in json_list if _.endswith(".jsonl")])
    total_file_nums = len(json_list)
    all_indices = list(range(total_file_nums))
    seg_nums = 8
    indices = [all_indices[_::seg_nums] for _ in range(seg_nums)]
    args = parse_args()
    run(save_dir=args.save_dir, indices=indices[args.seg_index])
