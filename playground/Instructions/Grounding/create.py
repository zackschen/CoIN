import os
import json,random
import numpy as np

# ################################################################
dirs = os.listdir('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding')

multi_datas = []

for dir_ in dirs:
    if 'train' in dir_:
        train_path = os.path.join('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding',dir_)
        if os.path.exists(train_path):
            datas = json.load(open(train_path))
            multi_datas.extend(datas)
            print('Append Train: %s'%(train_path))

random.shuffle(multi_datas)

json.dump(multi_datas,open('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding/train.json','w'),indent=4)



test_multi_datas = []
for dir_ in dirs:
    if 'test' in dir_ and 'create' not in dir_:
        train_path = os.path.join('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding',dir_)
        if os.path.exists(train_path):
            datas = json.load(open(train_path))
            test_multi_datas.extend(datas)
            print('Append Test: %s'%(train_path))

random.shuffle(test_multi_datas)

json.dump(test_multi_datas,open('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding/test.json','w'),indent=4)