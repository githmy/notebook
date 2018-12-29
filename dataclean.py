import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
import json
import csv
import numpy as np
import logging
import os
import sys

TXTPATH = "./data"


def json2csv(jsonfin, csvout):
    # 1. 读文件
    columobj = ['comment_text']
    with open(jsonfin, encoding='utf-8') as json_file:
        setting = json.loads(json_file.read())
        setts = setting['rasa_nlu_data']['common_examples']
        # 2. 读出所有意图
        for i1 in setts:
            if i1['intent'] not in columobj:
                columobj.append(i1['intent'])
        # 3. 根据标签设布尔值
        csv_pd = pd.DataFrame(data={}, columns=columobj)
        csv_pd.set_index(['comment_text'], inplace=True)
        columobj.pop(0)
        # print(len(columobj))
        # exit(0)
        for i1 in setts:
            tmpjson = {}
            for i2 in columobj:
                if i1['intent'] == i2:
                    tmpjson.__setitem__(i2, 1)
                else:
                    tmpjson.__setitem__(i2, 0)
            csv_pd.loc[i1['text']] = tmpjson
        # 3. 写入csv文件
        csv_pd.to_csv(csvout)
    return True


if __name__ == '__main__':
    jsonfin = os.path.join(TXTPATH, "rasa_zh_root.json")
    csvout = os.path.join(TXTPATH, "rasa_train.csv")
    json2csv(jsonfin, csvout)
