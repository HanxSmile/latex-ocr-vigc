import os
import json
from threading import Thread
from collections import defaultdict
import re
import tqdm
import time


def find_doc(work_dir, file_name, save_dir, unsafe_words):
    print(f'Start to open {file_name}')
    with open(os.path.join(work_dir, file_name), 'r') as f:
        xiaohongshu = f.readlines()

    final_data = []
    file_len = len(xiaohongshu)
    print(f'{file_name} len is {file_len}')
    BLOCK_block_dict = defaultdict(int)
    for data in tqdm.tqdm(xiaohongshu, desc=f"Procesing {file_name}"):
        data = json.loads(data)
        content = data['content']
        # print(content)
        for hit_word in unsafe_words:
            result_search = re.search(hit_word, content)
            # print(result_search)
            if result_search:
                BLOCK_block_dict[hit_word] += 1
                data['high_hit_word'] = hit_word
                final_data.append(data)
                break

    text = ""
    for key, item in BLOCK_block_dict.items():
        text += f"{key}: {item}\n"

    with open(os.path.join(save_dir, file_name.split('.')[0] + '_hits.json'), 'w', encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False)

    with open(os.path.join(save_dir, file_name.split('.')[0] + '_hits_statistic.txt'), 'w', encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    work_dir = '/mnt/petrelfs/share_data/wangbin/mllm/sz_unsafe/xiaohongshu/'
    json_list = os.listdir(work_dir)
    save_dir = '/mnt/petrelfs/ouyanglinke/clean_data/xiaohongshu/'

    with open('/mnt/petrelfs/share_data/ouyanglinke/clean_data/common/high.jsonl', 'r') as f:
        hit_words = f.readlines()

    unsafe_words = []
    for hit in hit_words:
        hit = json.loads(hit)
        unsafe_words.append(hit['word'])

    find_doc(work_dir, json_list[0], save_dir, unsafe_words)
    # t0 = Thread(target=find_doc, args=(work_dir, json_list, save_dir, unsafe_words))
    # t1 = Thread(target=find_doc, args=(work_dir, json_list, save_dir, unsafe_words))
    # t2 = Thread(target=find_doc, args=(work_dir, json_list[2], save_dir))
    # t3 = Thread(target=find_doc, args=(work_dir, json_list[3], save_dir))
    # t4 = Thread(target=find_doc, args=(work_dir, json_list[4], save_dir))
    # t5 = Thread(target=find_doc, args=(work_dir, json_list[5], save_dir))
    # t6 = Thread(target=find_doc, args=(work_dir, json_list[6], save_dir))
    # t7 = Thread(target=find_doc, args=(work_dir, json_list[7], save_dir))
    # t8 = Thread(target=find_doc, args=(work_dir, json_list[8], save_dir))
    # t9 = Thread(target=find_doc, args=(work_dir, json_list[9], save_dir))

    # 启动线程运行
    t0.start()
    t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t5.start()
    # t6.start()
    # t7.start()
    # t8.start()
    # t9.start()

    # 等待所有线程执行完毕,join() 等待线程终止，要不然一直挂起
    t0.join()
    t1.join()
    # t2.join()
    # t3.join()
    # t4.join()
    # t5.join()
    # t6.join()
    # t7.join()
    # t8.join()
    # t9.join()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Finished")