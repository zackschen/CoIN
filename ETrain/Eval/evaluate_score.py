import json, os, argparse, re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./results/CoIN/Qwen_Chat/ScienceQA/Finetune")
    return parser.parse_args()

def is_last_layer_folder(path):
    file_list = os.listdir(path)
    for file in file_list:
        if os.path.isdir(os.path.join(path,file)):
            return False
    return True

if __name__ == '__main__':

    args = parse_args()

    path = os.path.join(args.dir,"prompt_eval_merge.jsonl")
    prompts = []
    with open(path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    total = 0
    scores = 0
    for prompt in prompts:
        total += 10
        score = prompt['text']
        score.isnumeric()
        try:
            score = float(score)
            if score <= 10.0:
                scores += score
        except:
            if '/' in score:
                score = score.split('/')[0]
                if score.isnumeric() and float(score) <= 10.0:
                    scores += float(score)
            else:
                pattern = re.compile(r'[1-9]')
                res = pattern.findall(score)
                if len(res) > 0:
                    scores += float(res[0])
                else:
                    print(score)
        
    
    final_score = scores / total

    output_file = os.path.join(args.dir, 'Prompt_Result.text')
    with open(output_file, 'w') as f:
        f.write('Final score:{}'.format(round(final_score*100.0)))