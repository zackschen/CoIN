import os
import json, random
import numpy as np
# ################################################################

MemoryError_datas = []
Dataset = 'ImageNet'
splits = ['train','train_new']
number_of_memory = [100,500,1000]

for split in splits:
    for number in number_of_memory:
        train_path = os.path.join('./playground/Instructions/{}/{}.json'.format(Dataset,split))
        assert os.path.exists(train_path)
        if os.path.exists(train_path):
            datas = json.load(open(train_path))

        length = number
        memory_choice = np.random.choice(len(datas),length,replace=False)
        memory_choice = [datas[i] for i in memory_choice]

        random.shuffle(memory_choice)

        json.dump(memory_choice,open('./playground/Instructions/{}/{}_memory_{}.json'.format(Dataset,split,number),'w'),indent=4)