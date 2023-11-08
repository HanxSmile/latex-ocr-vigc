import sys
import time
import json
import requests
import tqdm
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


def call(d: dict, output_file: str):
    # print(f"processing {idx} ...", file=sys.stderr)
    while True:
        result = req(d)
        if not result:
            time.sleep(1)
            continue
        break

    # print(result)
    # result.pop('auxInfo')
    d["nlp_content_risk_result"] = result
    d['content_risk_provider'] = "数美"

    with open(output_file, 'a',  encoding="utf8") as f:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")


    # output_lock.acquire()
    # print(json.dumps(d, ensure_ascii=False))
    # output_lock.release()


# from concurrent.futures import ThreadPoolExecutor

# with ThreadPoolExecutor(max_workers=10) as pool:
#     # with open("ml-cc-10dump-rush_dedup_v001_zh_CC-MAIN-2023-06.jsonl", "r") as f:
#     with open("ishumei/test_result.jsonl", "r") as f:
#         lines = f.readlines()
#         for idx, line in enumerate(lines):
#             pool.submit(call, idx, json.loads(line))
#     pool.shutdown()

file_path = sys.argv[1]
output_file = file_path.split('.')[0] + '_shumei.jsonl'

with open(file_path, "r") as f:
    # dir_list = json.load(f)
    dir_list = [json.loads(s) for s in f.readlines()]
    # dir_list = data['hits']['hits']

end_idx = len(dir_list)
if os.path.exists(output_file):
    with open(output_file, 'r') as ff:
        finished = ff.readlines()
        start_idx = len(finished)
        check_point = json.loads(finished[-1])
        match_point = dir_list[start_idx-1]
        if check_point['id'] != match_point['id']:
            print(f"The check point {check_point} not match the match point {match_point}, the start id should not be {start_idx}. Check.")
            sys.exit()
else:
    start_idx = 0  # 开始的index

print(f'The process for {file_path} will start from {start_idx} to {end_idx}, the output file is {output_file}.')
for sample in tqdm.tqdm(dir_list[start_idx:end_idx]):
    # print(line['content'])
    # docs = sample['_source']
    call(sample, output_file)