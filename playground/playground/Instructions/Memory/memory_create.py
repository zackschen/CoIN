import os
import json, random
import numpy as np
# ################################################################

dirs = os.listdir('./playground/Instructions')


number_of_memory = [100,500,1000]
Stage_list = ['ScienceQA','TextVQA','ImageNet','GQA','VizWiz','Grounding','VQAv2','OCRVQA']

for number in number_of_memory:
    for stage in range(1,len(Stage_list)):
        memory_datas = []
        stage_datas = Stage_list[:stage]
        print('Stage: {};\t Combine: {}'.format(Stage_list[stage],stage_datas))
        for dir_ in dirs:
            if dir_ not in stage_datas:
                continue
            memory_path = os.path.join('./playground/Instructions',dir_,'train_memory_{}.json'.format(number))
            assert os.path.exists(memory_path)
            if os.path.exists(memory_path):
                datas = json.load(open(memory_path))
                memory_datas.extend(datas)
                print('Append %s'%(memory_path))

        random.shuffle(memory_datas)

        json.dump(memory_datas,open('./playground/Instructions/Memory/{}_memory_{}.json'.format(Stage_list[stage],number),'w'),indent=4)