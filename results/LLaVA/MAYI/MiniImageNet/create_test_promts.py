import json, os

answers = open(os.path.expanduser('./results/MiniImageNet/Finetune/miniimagenet_answers.jsonl'))
promts_answers = []

for i, ans_js in enumerate(answers):
    ans = json.loads(ans_js)
    text = ans['text']
    label = ans['label']

    system_dict = {"role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",}
    user_dict = {"role": "user",
    "content": f"Please only answer the question in yes or no. Is the \"Prediction\" correctly predicting the right \"Label\"? Label: {label}, Prediction: {text}"}
    
    promts_answers.append([system_dict, user_dict])

json.dump(promts_answers,open('./results/MiniImageNet/Finetune/miniimagenet_answers_prompts.jsonl','w'),indent=4)