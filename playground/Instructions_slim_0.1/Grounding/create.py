import os
import json
import numpy as np

# ################################################################
dirs = os.listdir('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding')

multi_datas = []

for dir_ in dirs:
    if 'train_refcoco' in dir_:
        train_path = os.path.join('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding',dir_)
        if os.path.exists(train_path):
            datas = json.load(open(train_path))
            multi_datas.extend(datas)
            print('Append %s'%(train_path))

# json.dump(multi_datas,open('/home/chencheng/Code/LLaVA/playground/Instructions/Grounding/train.json','w'),indent=4)

for percent in [0.1,0.2,0.4,0.6,0.8]:
    dir_path = f'/home/chencheng/Code/LLaVA/playground/Instructions_slim_{percent}/Grounding'
    os.makedirs(dir_path,exist_ok=True)

    train_length = int(len(multi_datas)*percent)
    train_choice = np.random.choice(len(multi_datas),train_length,replace=False)
    train_choice = [multi_datas[i] for i in train_choice]
    json.dump(train_choice,open(f'/home/chencheng/Code/LLaVA/playground/Instructions_slim_{percent}/Grounding/train.json','w'),indent=4)
