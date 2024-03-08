import json, os, argparse, re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./results/CLIT_normaltrain_testslim/ImageNet/OCRVQA")
    return parser.parse_args()

def is_last_layer_folder(path):
    file_list = os.listdir(path)
    for file in file_list:
        if os.path.isdir(os.path.join(path,file)):
            return False
    return True

if __name__ == '__main__':

    args = parse_args()
    dirs = os.listdir(args.dir)

    prompt_dirs = []
    pattern = re.compile(r'prompt_[1-9]')
    
    for dir_ in dirs:
        res = pattern.findall(dir_)
        if len(res) > 0:
            prompt_dirs.append(os.path.join(args.dir,dir_))
    
    prompts = []
    for path in prompt_dirs:
        prompt = [json.loads(line) for line in open(path)]
        prompts += prompt
    
    total = 0
    scores = 0
    scores_ten = []
    for prompt in prompts:
        total += 10
        score = prompt['text']
        try:
            score = float(score)
            if score <= 10.0:
                scores += score
                if score == 10.0:
                    scores_ten.append(prompt) 
        except:
            if '/' in score:
                score = score.split('/')[0]
                if score.isnumeric() and float(score) <= 10.0:
                    scores += float(score)
                    if float(score) == 10.0:
                        scores_ten.append(prompt) 
            else:
                print(score)
        
    
    final_score = scores / total

    output_file = os.path.join(args.dir, 'Prompt_Result.text')
    with open(output_file, 'w') as f:
        f.write('Final score:{}'.format(round(final_score*100.0)))
        json.dump(scores_ten,f,indent=4)