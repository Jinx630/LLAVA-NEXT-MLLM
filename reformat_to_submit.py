"""
@Time: 2024/8/25 23:14
@Author: xujinlingbj
@File: reformat_to_submit.py
"""
import json
import sys
from collections import defaultdict

def load_jsonl_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = json.loads(data[i])
    return data


def save_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def reformat_to_submit(data_path, save_path):   
    data = load_jsonl_data(data_path)
    res_data = []
    keywords_to_example = defaultdict(list)
    for line in data:
        temp_dic = {'index': line['origin_index'], 'model_answer': [line['predict']]}

        keywords_to_example[line['keyword']].append(temp_dic)
    for key, value in keywords_to_example.items():
        value = sorted(value, key=lambda x: x['index'])
        res_data.append({'keyword': key, 'example': value})

    save_json_file(save_path, res_data)


if __name__ == '__main__':
    data_path=sys.argv[1]
    save_path=sys.argv[2]
    reformat_to_submit(data_path, save_path)