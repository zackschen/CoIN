import pickle,os,json
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

info = unpickle('/mnt/hdd2/chencheng/cl_dataset/CIFAR100/cifar-100-python/meta')
labels = info['fine_label_names']
labels_dict = dict(zip(range(len(labels)), labels))

all_labels_string = ','.join(labels)

train_path = './cl_dataset/CIFAR100_withlabel/train/'
test_path = './cl_dataset/CIFAR100_withlabel/test/'

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

another_cadidate_promts = ["Provide a detailed description of the subject in the image.",
                   "Specify the classification of the object captured in the image. ",
                   "In a single word, describe the category of the object in the image.",
                   "In this image, what is the primary category of the object? ",
                   "Assign a single-word label to categorize the object within the image.",
                   "Offer a classification for the item visible in the image. ",
                   "Describe the class or type that best characterizes the object in the image.",
                   "Name the category that best describes the object in this image.",
                   "Identify and categorize the primary subject matter in the image.",
                   "Distinguish the type of object represented in the given image.",]

# promts = []

# data = unpickle('/mnt/hdd2/chencheng/cl_dataset/CIFAR100/cifar-100-python/train')
# for i in range(len(data['filenames'])):
#     dir_path = os.path.join(train_path, labels_dict[data['fine_labels'][i]])
#     img_name = os.path.join(dir_path, data['filenames'][i])

#     promt_dict = {}
#     promt_dict['id'] = data['filenames'][i]
#     promt_dict['image'] = img_name
#     convers = []
#     conver1 = {}
#     conver1['from'] = 'human'
#     conver1['value'] = cadidate_promts[np.random.randint(0,10)] + " <image>" 

#     label = labels_dict[data['fine_labels'][i]]
#     conver2 = {}
#     conver2['from'] = 'gpt'
#     conver2['value'] = "The object is a {}.".format(label)
#     convers.append(conver1)
#     convers.append(conver2)

#     promt_dict['conversations'] = convers

#     promts.append(promt_dict)

# json.dump(promts,open('playground/data/CIFAR100/cifar100_train_diversity10.json','w'),indent=4)

questions = []

data = unpickle('/mnt/hdd2/chencheng/cl_dataset/CIFAR100/cifar-100-python/test')
for i in range(len(data['filenames'])):
    dir_path = os.path.join(test_path, labels_dict[data['fine_labels'][i]])
    img_name = os.path.join(dir_path, data['filenames'][i])
    label = labels_dict[data['fine_labels'][i]]

    question = {}
    question['question_id'] = str(i)
    question['image'] = img_name
    question['text'] = "{} Please only answer a single object in {}.".format(another_cadidate_promts[np.random.randint(0,10)],all_labels_string)
    question['answer'] = label.capitalize()
    questions.append(question)

json.dump(questions,open('playground/data/CIFAR100/cifar100_question_diversity10_another.json','w'),indent=4)