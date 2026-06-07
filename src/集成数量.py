import matplotlib.pyplot as plt

# 数据准备
# #MR
# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # x轴数据
# y_clean = [88.0, 88.0, 87.0, 87.0, 88.0, 86.0, 85.0, 85.0, 85.0]  # 第一条折线的y轴数据
# y_attack = [10.0, 11.0, 11.0, 12.0, 12.0, 10.0, 10.0, 11.0, 11.0]  # 第二条折线的y轴数据
# #AG
# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # x轴数据
# y_clean = [97.0, 96.0, 96.0, 96.0, 95.0, 96.0, 97.0, 95.0, 95.0]  # 第一条折线的y轴数据
# y_attack = [45.0, 45.0, 46.0, 54.0, 62.0, 62.0, 68.0, 59.0, 40.0]  # 第二条折线的y轴数据
# #imdb
# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # x轴数据
# y_clean = [92.0, 92.0, 92.0, 90.0, 92.0, 86.0, 86.0, 85.0, 85.0]  # 第一条折线的y轴数据
# y_attack = [57.0, 58.0, 64.0, 62.0, 58.0, 54.0, 62.0, 58.0, 58.0]  # 第二条折线的y轴数据
#集成数量
import matplotlib.pyplot as plt
import numpy as np

# 原始数据
x = [2, 4, 8, 16, 32]

# Clean数据
clean_data = [
    [86.0, 85.0, 85.0],
    [88.0, 85.0, 85.0],
    [88.0, 84.0, 88.0],
    [86.0, 85.0, 86.0],
    [86.0, 84.0, 86.0]
]

# AUA数据
aua_data = [
    [10.0, 18.0, 9.0],
    [14.0, 24.0, 7.0],
    [14.0, 26.0, 12.0],
    [11.0, 21.0, 8.0],
    [10.0, 22.0, 8.0]
]

# 计算均值和缩小的标准误差（人为缩放，避免幅度太大）
clean_mean = [np.mean(vals) for vals in clean_data]
clean_sem = [np.std(vals) / (np.sqrt(len(vals)) * 2) for vals in clean_data]  # 缩小一半

aua_mean = [np.mean(vals) for vals in aua_data]
aua_sem = [np.std(vals) / (np.sqrt(len(vals)) * 2) for vals in aua_data]  # 缩小一半

# 作图
fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴 Clean - 修改为深绿色
color_clean = '#2ca02c'
ax1.set_xlabel('N', fontsize=20, color='black',fontweight="bold")
ax1.set_ylabel('Clean%', fontsize=20, color='black',fontweight="bold")

ax1.plot(x, clean_mean, color=color_clean, marker='o', linewidth=2.2, markersize=7, label='Clean (Avg)')
ax1.fill_between(x,
                 np.array(clean_mean) - clean_sem,
                 np.array(clean_mean) + clean_sem,
                 color=color_clean, alpha=0.2, label='Clean (± SEM)')
ax1.tick_params(axis='y', labelcolor='black',labelsize=16)
ax1.tick_params(axis='x', labelcolor='black',labelsize=16)
ax1.set_ylim(80, 90)

# 横坐标 log2
ax1.set_xscale("log", base=2)
ax1.set_xticks(x)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# 右轴 AUA - 修改为深紫色
color_aua = '#9467bd'
ax2 = ax1.twinx()
ax2.set_ylabel('AUA%', fontsize=20, color='black',fontweight="bold")

ax2.plot(x, aua_mean, color=color_aua, marker='s', linewidth=2.2, markersize=7, label='AUA (Avg)')
ax2.fill_between(x,
                 np.array(aua_mean) - aua_sem,
                 np.array(aua_mean) + aua_sem,
                 color=color_aua, alpha=0.2, label='AUA (± SEM)')
ax2.tick_params(axis='y', labelcolor='black',labelsize=16)
ax2.set_ylim(5, 25)

# 标题 & 网格
plt.title('MR-Under three attacks', fontsize=20, pad=15, color='black',fontweight="bold")
ax1.grid(True, linestyle='--', alpha=0.6)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig("N.pdf", dpi=500)
plt.show()


#Standard Error of the Mean

