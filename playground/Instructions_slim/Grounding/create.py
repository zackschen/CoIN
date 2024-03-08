import os
import json
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
            print('Append %s'%(train_path))

# json.dump(multi_datas,open('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding/train.json','w'),indent=4)

length = 10000
train_choice = np.random.choice(len(multi_datas),length,replace=False)
train_choice = [multi_datas[i] for i in train_choice]

# json.dump(train_choice,open('/home/chencheng/Code/LLaVA/playground/Instructions_slim/Grounding/train.json','w'),indent=4)

test_multi_datas = []
for dir_ in dirs:
    if 'train' not in dir_ and 'create' not in dir_:
        train_path = os.path.join('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding',dir_)
        if os.path.exists(train_path):
            datas = json.load(open(train_path))
            test_multi_datas.extend(datas)
            print('Append Test: %s'%(train_path))

length = 4000
test_choice = np.random.choice(len(test_multi_datas),length,replace=False)
test_choice = [test_multi_datas[i] for i in test_choice]
json.dump(test_choice,open('/home/chencheng/Code/LLaVA/playground/Instructions_slim/Grounding/test.json','w'),indent=4)