import re
import os
import json
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class HitWordDataset(Dataset):
    def __init__(self, work_dir, unsafe_words_anno_path, indices=None):
        with open(unsafe_words_anno_path, 'r') as f:
            hit_words = f.readlines()

        json_list = sorted(os.listdir(work_dir))

        unsafe_words = []
        for hit in hit_words:
            hit = json.loads(hit)
            unsafe_words.append(hit['word'])
        self.unsafe_words = unsafe_words
        self.compile_unsafe_words = [re.compile(_) for _ in self.unsafe_words]

        if indices is not None:
            if isinstance(indices, str):
                with open(indices) as indices_f:
                    indices = json.load(indices_f)
            json_list = [json_list[_] for _ in indices]
        self.inner_dataset_names = []
        self.inner_dataset_contents = []
        self.inner_dataset_nums = []
        self.inner_dataset_cumnums = []
        for file in tqdm(json_list):
            anno_path = os.path.join(work_dir, file)
            with open(os.path.join(anno_path), 'r') as f:
                this_data = f.readlines()
                self.inner_dataset_contents.append(this_data)
                self.inner_dataset_nums.append(len(this_data))
                self.inner_dataset_names.append(file)
                self.inner_dataset_cumnums.append(sum(self.inner_dataset_nums))

    def __len__(self):
        return sum([len(_) for _ in self.inner_dataset_contents])

    def _get_index(self, index):
        for i, num in enumerate(self.inner_dataset_cumnums):
            if index >= num:
                continue
            if i == 0:
                return i, index
            else:
                return i, index - num

    def __getitem__(self, index):
        ds_id, data_id = self._get_index(index)
        file_name = self.inner_dataset_names[ds_id]
        data = json.loads(self.inner_dataset_contents[ds_id][data_id])["content"]
        res = {"raw_data": data, "file_name": file_name, "index": data_id, "uid": f"{file_name}_{data_id}",
               "high_hit_word": None}
        for hit_word, compile_unsafe_word in zip(self.unsafe_words, self.compile_unsafe_words):
            result_search = re.search(compile_unsafe_word, data["content"])
            # print(result_search)
            if result_search:
                res['high_hit_word'] = hit_word
                break
        return res

    def collater(self, batch):
        return batch
