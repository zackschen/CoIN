import pickle,os,json
import numpy as np
from PIL import Image
import cv2, shutil

dict = json.load(open('/mnt/hdd2/chencheng/cl_dataset/LLaVA/MiniImageNet_withlabel/imagenet_class_index.json'))

new_dict = {}

for key, value in dict.items():
    new_dict[value[0]] = value[1]

path = "/mnt/hdd2/chencheng/cl_dataset/LLaVA/MiniImageNet_withlabel/train/"
val_path = "/mnt/hdd2/chencheng/cl_dataset/LLaVA/MiniImageNet_withlabel/val/"

cadidate_promts = ["What is the object in the image? ",
                   "Classify the content of the image into its corresponding type.",
                   "Elaborate on how you would categorize the object in the image.",
                   "which is the catagory of the object in the image?",
                   "Classify the image?",
                   "How to describe the catasgory of the object in the image?",
                   "There is a object in this image, what is the catagory of it?",
                   "Please usa a word to classify the obejct in the image.",
                   "Identify the item depicted in the image.",
                   "Determine the specific category of the object in the image.",]

files= os.listdir(path)
count = 0
promts = []
questions = []

labels = []
for file in files:
     if os.path.isdir(os.path.join(path,file)):
        # labels.append(new_dict[file])
        labels.append(file)
all_labels_string = ','.join(labels)

for file in files:
     if os.path.isdir(os.path.join(path,file)):
        
        sub_files= os.listdir(os.path.join(path,file))
        
        # for sub_file_name in sub_files[:-100]:
        for sub_file_name in sub_files:
            promt_dict = {}
            promt_dict['id'] = sub_file_name
            # promt_dict['image'] = os.path.join(path,new_dict[file],sub_file_name)
            promt_dict['image'] = os.path.join(path,file,sub_file_name)
            convers = []
            conver1 = {}
            conver1['from'] = 'human'
            conver1['value'] = cadidate_promts[np.random.randint(0,10)] + " <image>" 

            # label = new_dict[file]
            label = file
            conver2 = {}
            conver2['from'] = 'gpt'
            conver2['value'] = "The object is a {}.".format(label)
            convers.append(conver1)
            convers.append(conver2)

            promt_dict['conversations'] = convers

            promts.append(promt_dict)

        count += 1
        # os.makedirs(os.path.join(val_path,new_dict[file]),exist_ok=True)

        # for sub_file_name in sub_files[-100:]:
        #     origin_path = os.path.join(path,file,sub_file_name)
        #     file_path = os.path.join(val_path,new_dict[file],sub_file_name)

        #     label = new_dict[file]
        #     question = {}
        #     question['question_id'] = sub_file_name
        #     question['image'] = file_path
        #     question['text'] = "{} Please only answer a single object in {}.".format(cadidate_promts[np.random.randint(0,10)],all_labels_string)
        #     question['answer'] = label.capitalize()
        #     questions.append(question)

        #     shutil.move(origin_path,file_path)

        # os.rename(os.path.join(path,file),os.path.join(path,new_dict[file]))
        # print(count)

json.dump(promts,open('playground/data/MiniImageNet/miniimagenet_train_diversity10.json','w'),indent=4)
# json.dump(questions,open('playground/data/MiniImageNet/miniimagenet_question.json','w'),indent=4)
        
