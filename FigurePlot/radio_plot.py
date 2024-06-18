import matplotlib.pyplot as plt
import numpy as np

# 数据准备

#Qwen-VL
categories = ['ImageNet','TextVQA','ScienceQA','OCR-VQA', 'VQAV2', 'Grounding', 'VizWiz', 'GQA']
values_1 = [53.7, 66.36, 67.69, 47.8, 71, 63.1, 36.38, 59.3]
values_2 = [29.57, 42.45, 31.05, 47.8, 67.75, 40.33, 15.3, 55.57]

#LLaVA
# categories = ['ImageNet','TextVQA','ScienceQA','OCR-VQA', 'VQAV2', 'Grounding', 'VizWiz', 'GQA']
# values_1 = [96.05,      49.99,       82.45,     57.08,      62.2,   31.27,      55.45,      56.4]
# values_2 = [10.25,      28.74,       21.26,     57.08,      42.5,   0.83,      32.45,       36.78]

#Minigpt-v2
# categories = ['ImageNet','TextVQA','ScienceQA','OCR-VQA', 'VQAV2', 'Grounding', 'VizWiz', 'GQA']
# values_1 = [7.25,          10.4,     28.81,   6.15,      36.1,    0,          41.35,      31.55]
# values_2 = [11.9,      29.89,       44.35,    6.15,      38.1,   0,           42.58,      36.95]


colors = ['#219ebc','#d62828']

# 计算角度
num_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

# 绘制雷达图
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
values_1 += values_1[:1]  # 保持values的长度与angles一致
values_2 += values_2[:1]  # 保持values的长度与angles一致
angles += angles[:1]
ax.plot(angles, values_1, color=colors[0])
ax.fill(angles, values_1, color=colors[0], alpha=0.25)
ax.plot(angles, values_2, color=colors[1])
ax.fill(angles, values_2, color=colors[1], alpha=0.25)

# 添加标签和标题
ax.set_thetagrids(np.degrees(angles[:-1]), categories, size = 22)
ax.set_title('Qwen-VL', size = 30, color=colors[1])

# 显示图形
plt.savefig('./FigurePlot/Qwen-VL_radar.pdf')
plt.close()
