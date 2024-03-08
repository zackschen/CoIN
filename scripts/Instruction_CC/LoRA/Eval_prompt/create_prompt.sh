

python playground/create_prompt.py \
    --rule ./llava/eval/table/rule.json \
    --questions ./playground/Instructions_slim/Grounding/test.json \
    --results ./results/CLIT_slim_0.6/Grounding/VQAv2/merge.jsonl \
    --rule_temp CLIT_Grounding \

python playground/create_prompt.py \
    --rule ./llava/eval/table/rule.json \
    --questions ./playground/Instructions_slim/Grounding/test.json \
    --results ./results/CLIT_slim_0.6/Grounding/OCRVQA/merge.jsonl \
    --rule_temp CLIT_Grounding \
