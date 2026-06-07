import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

y1max = [88.0, 88.0, 87.0, 87.0, 88.0, 86.0, 85.0, 85.0, 85.0]
y1min = [87.0, 87.0, 86.0, 86.0, 87.0, 85.0, 84.0, 84.0, 84.0]
y1 = np.mean([y1max, y1min], axis=0)  # 均值
y1_sem = np.std([y1max, y1min], axis=0) / np.sqrt(2)  # 标准误

y2max = [10.0, 11.0, 11.0, 12.0, 12.0, 10.0, 10.0, 11.0, 11.0]
y2min = [7.0, 9.0, 10.0, 10.0, 11.0, 9.0, 9.0, 8.0, 8.0]
y2 = np.mean([y2max, y2min], axis=0)  # 均值
y2_sem = np.std([y2max, y2min], axis=0) / np.sqrt(2)  # 标准误

y3max = [97.0, 96.0, 96.0, 96.0, 95.0, 96.0, 98.0, 95.0, 95.0]
y3min = [96.0, 95.0, 95.0, 94.0, 94.0, 95.0, 97.0, 94.0, 93.0]
y3 = np.mean([y3max, y3min], axis=0)  # 均值
y3_sem = np.std([y3max, y3min], axis=0) / np.sqrt(2)  # 标准误

y4max = [45.0, 45.0, 46.0, 54.0, 62.0, 62.0, 68.0, 59.0, 40.0]
y4min = [42.0, 43.0, 44.0, 50.0, 60.0, 61.0, 65.0, 53.0, 38.0]
y4 = np.mean([y4max, y4min], axis=0)  # 均值
y4_sem = np.std([y4max, y4min], axis=0) / np.sqrt(2)  # 标准误

# 配色（科研友好，蓝+橙）
color1 = "#1f77b4"  # 蓝
color2 = "#d62728"  # 红橙
fill_alpha = 0.25   # 阴影透明度

# 创建图形
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), dpi=500)
plt.subplots_adjust(wspace=0.28)

# 子图 1: MR
ax1.set_ylim(82, 94)
ax1.set_xticks(x)
ax1.set_xlabel("r", fontsize=20, fontweight="bold")
ax1.set_ylabel("Clean%", fontsize=20, fontweight="bold", color='black')
ax1.tick_params(axis='both', labelsize=14)

# 阴影部分（标准误）
fill1 = ax1.fill_between(x, y1 - y1_sem, y1 + y1_sem,
                         facecolor=color1, alpha=fill_alpha,
                         label="Clean (± SEM)")
# 折线部分（均值）
line1, = ax1.plot(x, y1, marker='o', color=color1,
                  linewidth=2.2, markersize=6,
                  label="Clean (Avg)")

ax1_twin = ax1.twinx()
ax1_twin.set_ylim(0, 18)
ax1_twin.set_ylabel("AUA%", fontsize=18, fontweight="bold", color='black')
ax1_twin.tick_params(axis='y', labelsize=14)

# 阴影部分（标准误）
fill2 = ax1_twin.fill_between(x, y2 - y2_sem, y2 + y2_sem,
                              facecolor=color2, alpha=fill_alpha,
                              label="AUA (± SEM)")
# 折线部分（均值）
line2, = ax1_twin.plot(x, y2, marker='s', color=color2,
                       linewidth=2.2, markersize=6,
                       label="AUA (Avg)")

ax1.set_title("MR", fontsize=20, fontweight="bold")
# 合并图例，明确区分均值和标准误
ax1.legend([line1, fill1, line2, fill2],
           [line1.get_label(), fill1.get_label(),
            line2.get_label(), fill2.get_label()],
           loc='upper left', fontsize=12, frameon=False)

# 子图 2: AG’s News
ax2.set_ylim(92, 99)
ax2.set_xticks(x)
ax2.set_xlabel("r", fontsize=20, fontweight="bold")
ax2.set_ylabel("Clean%", fontsize=20, fontweight="bold", color='black')
ax2.tick_params(axis='both', labelsize=14)

# 阴影部分（标准误）
fill3 = ax2.fill_between(x, y3 - y3_sem, y3 + y3_sem,
                         facecolor=color1, alpha=fill_alpha,
                         label="Clean (± SEM)")
# 折线部分（均值）
line3, = ax2.plot(x, y3, marker='o', color=color1,
                  linewidth=2.2, markersize=6,
                  label="Clean (Avg)")

ax2_twin = ax2.twinx()
ax2_twin.set_ylim(35, 70)
ax2_twin.set_ylabel("AUA%", fontsize=20, fontweight="bold", color='black')
ax2_twin.tick_params(axis='y', labelsize=14)

# 阴影部分（标准误）
fill4 = ax2_twin.fill_between(x, y4 - y4_sem, y4 + y4_sem,
                              facecolor=color2, alpha=fill_alpha,
                              label="AUA (± SEM)")
# 折线部分（均值）
line4, = ax2_twin.plot(x, y4, marker='s', color=color2,
                       linewidth=2.2, markersize=6,
                       label="AUA (Avg)")

ax2.set_title("AG’s News", fontsize=20, fontweight="bold")
# 合并图例，明确区分均值和标准误
ax2.legend([line3, fill3, line4, fill4],
           [line3.get_label(), fill3.get_label(),
            line4.get_label(), fill4.get_label()],
           loc='upper left', fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig("r.pdf", dpi=500, bbox_inches="tight")
plt.show()
