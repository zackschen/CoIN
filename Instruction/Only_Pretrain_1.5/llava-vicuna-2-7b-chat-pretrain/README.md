---
inference: false
---

<br>
<br>

# LLaVA Model Card

This is a pretrained checkpoint, you can use it to instruct tune your multimodal models.

Check out the instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/README.md#visual-instruction-tuning)

## Model details

**Model type:**
LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data.
It is an auto-regressive language model, based on the transformer architecture.

**Model date:**
LLaVA-v1.5-MLP2x-336px-Pretrain-Vicuna-7B-v1.5 was trained in September 2023.

**Paper or resources for more information:**
https://llava-vl.github.io/

## License
Llama 2 is licensed under the LLAMA 2 Community License, 
Copyright (c) Meta Platforms, Inc. All Rights Reserved.

**Where to send questions or comments about the model:**
https://github.com/haotian-liu/LLaVA/issues

## Intended use
**Primary intended uses:**
The primary use of LLaVA is research on large multimodal models and chatbots.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

## Training dataset
- 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP.
