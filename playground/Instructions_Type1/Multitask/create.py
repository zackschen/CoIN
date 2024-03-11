import os
import json, random

# ################################################################
dirs = os.listdir('./playground/Instructions')

# multi_datas = []

# for dir_ in dirs:
#     if dir_ == 'Multitask':
#         continue
#     train_path = os.path.join('./playground/Instructions',dir_,'train.json')
#     if os.path.exists(train_path):
#         datas = json.load(open(train_path))
#         multi_datas.extend(datas)
#         print('Append %s'%(train_path))

# random.shuffle(multi_datas)

# json.dump(multi_datas,open('./playground/Instructions/Multitask/train.json','w'),indent=4)



multi_datas = []

for dir_ in dirs:
    if dir_ == 'Multitask':
        continue
    train_path = os.path.join('./playground/Instructions',dir_,'train.json')
    train_new_path = os.path.join('./playground/Instructions',dir_,'train_new.json')
    if os.path.exists(train_new_path):
        train_path = train_new_path
    if os.path.exists(train_path):
        datas = json.load(open(train_path))
        multi_datas.extend(datas)
        print('Append %s'%(train_path))

random.shuffle(multi_datas)

json.dump(multi_datas,open('./playground/Instructions/Multitask/train_new.json','w'),indent=4)