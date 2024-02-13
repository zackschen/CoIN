import pickle,os,json
import numpy as np
from PIL import Image
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
info = unpickle('/mnt/hdd3/chencheng/cl_dataset/CIFAR10/cifar-10-batches-py/batches.meta')
labels = info['label_names']
labels_dict = dict(zip(range(len(labels)), labels))

all_labels_string = ','.join(labels)

train_path = './cl_dataset/CIFAR10_withlabel/train/'
test_path = './cl_dataset/CIFAR10_withlabel/test/'

# promts = []

# for j in range(1,6):
#     data_i = unpickle('/mnt/hdd3/chencheng/cl_dataset/CIFAR10/cifar-10-batches-py/data_batch_'+str(j))
#     data_labels = data_i['labels']
#     data_datas = data_i['data']
#     data_filenames = data_i['filenames']

#     for i in range(len(data_filenames)):
#         dir_path = os.path.join(train_path, labels_dict[data_labels[i]])
#         img_name = os.path.join(dir_path, data_filenames[i])

#         os.makedirs(dir_path,exist_ok=True)

#         img = np.reshape(data_datas[i], (3, 32, 32))
#         img = img.transpose(1, 2, 0)
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         cv2.imwrite(img_name,img)


#         promt_dict = {}
#         promt_dict['id'] =data_filenames[i]
#         promt_dict['image'] = img_name
#         convers = []
#         conver1 = {}
#         conver1['from'] = 'human'
#         conver1['value'] = "What is the object in the image? <image>"

#         label = labels_dict[data_labels[i]]
#         conver2 = {}
#         conver2['from'] = 'gpt'
#         conver2['value'] = "The object is a {}.".format(label)
#         convers.append(conver1)
#         convers.append(conver2)

#         promt_dict['conversations'] = convers

#         promts.append(promt_dict)

# json.dump(promts,open('playground/data/cifar10.json','w'),indent=4)

questions = []

test_data = unpickle('/mnt/hdd3/chencheng/cl_dataset/CIFAR10/cifar-10-batches-py/test_batch')

for i in range(len(test_data['filenames'])):
    dir_path = os.path.join(test_path, labels_dict[test_data['labels'][i]])
    img_name = os.path.join(dir_path, test_data['filenames'][i])

    os.makedirs(dir_path,exist_ok=True)

    img = np.reshape(test_data['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name,img)


    label = labels_dict[test_data['labels'][i]]

    question = {}
    question['question_id'] = str(i)
    question['image'] = img_name
    question['text'] = "What is the object in the image? Please only answer a single object in {}.".format(all_labels_string)
    question['answer'] = label.capitalize()
    questions.append(question)
    
json.dump(questions,open('playground/data/cifar10_question.json','w'),indent=4)