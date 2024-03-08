import os,json

path = 'playground/Instructions/VQAv2/train.json'

# Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n

splits = path.split('/')
splits[1] = splits[1] + '_Qwen'
qwen_path = '/'.join(splits)

with open(os.path.expanduser(path), "r") as f:
    instructions = json.load(f)

qwen_instructions = []

for instruct in instructions:
    conversations = instruct['conversations']
    for conversation in conversations:
        role = conversation['from']
        
        if role == 'human':
            conversation['from'] = 'user'
            content = conversation['value']
            if '<image>' in content:
                image = instruct['image']
                if './' in image:
                    image = image[2:]
                image = os.path.join('./cl_dataset',image)
                content = content.replace('<image>', 'Picture 1: <img>{}</img>'.format(image))
            conversation['value'] = content
        else:
            conversation['from'] = 'assistant'
    instruct['conversations'] = conversations
    qwen_instructions.append(instruct)

os.makedirs(os.path.dirname(os.path.expanduser(qwen_path)), exist_ok=True)

with open(os.path.expanduser(qwen_path), "w") as f:
    json.dump(qwen_instructions, f, indent=4)