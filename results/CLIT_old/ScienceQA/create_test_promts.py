import json, os

rule_dict = json.load(open(os.path.expanduser('llava/eval/table/rule.json'), 'r'))

answers = open(os.path.expanduser('./results/CLIT_slim/ScienceQA/Finetune/merge.jsonl'))
questions = json.load(open(os.path.expanduser('./playground/Instructions/ScienceQA/test.json')))

promts_answers = []

for i, ans_js in enumerate(answers):
    ans = json.loads(ans_js)
    text = ans['text']
    question = questions[i]
    label = question['answer']

    system_dict = {"role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",}
    user_dict = {"role": "user",
    "content": f"Please only answer the question in yes or no. Is the \"Prediction\" correctly predicting the right \"Label\"? Label: {label}, Prediction: {text}"}
    
    promts_answers.append([system_dict, user_dict])

json.dump(promts_answers,open('./results/CIFAR100/Original/cifar100_answers_eval_diversity10_another_prompts.jsonl','w'),indent=4)