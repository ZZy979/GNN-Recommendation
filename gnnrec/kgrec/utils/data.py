import json


def iter_json(filename):
    """遍历每行一个JSON格式的文件。"""
    with open(filename, encoding='utf8') as f:
        for line in f:
            yield json.loads(line)
