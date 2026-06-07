
import matplotlib.pyplot as plt

datasets = [
    {
        'title': 'MR',
        'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'y_clean': [88.0, 88.0, 87.0, 87.0, 88.0, 86.0, 85.0, 85.0, 85.0],
        'y_attack': [10.0, 11.0, 11.0, 12.0, 12.0, 10.0, 10.0, 11.0, 11.0]
    },
    {
        'title': 'AG’s News',
        'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'y_clean': [97.0, 96.0, 96.0, 96.0, 95.0, 96.0, 97.0, 95.0, 95.0],
        'y_attack': [45.0, 45.0, 46.0, 54.0, 62.0, 62.0, 68.0, 59.0, 40.0]
    },
    {
        'title': 'imdb',
        'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'y_clean': [92.0, 92.0, 92.0, 90.0, 92.0, 86.0, 86.0, 85.0, 85.0],
        'y_attack': [57.0, 58.0, 64.0, 62.0, 58.0, 54.0, 62.0, 58.0, 58.0]
    },
    {
        'title': 'Yelp',
        'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'y_clean': [98.0, 98.0, 98.0, 98.0, 98.0, 96.0, 96.0, 95.0, 93.0],
        'y_attack': [46.0, 61.0, 61.0, 63.0, 65.0, 58.0, 63.0, 58.0, 45.0]
    }
]

# 创建子图，每行显示一个子图
fig, axs = plt.subplots(ncols=len(datasets), figsize=(25, 5), dpi=500)

for i, data in enumerate(datasets):
    title = data['title']
    x = data['x']
    y_clean = data['y_clean']
    y_attack = data['y_attack']

    # 绘制第一条折线（左边的y轴）

    if i==0:
        ax1 = axs[i]
        ax1.plot(x, y_clean, marker='o', linestyle='-', color='b', label='Clean%')
        ax1.set_xlabel('r', fontsize=30)
        #ax1.set_ylabel('Clean Accuracy', fontsize=20)
        ax1.tick_params(axis='both', labelsize=20)
        ax1.set_xticks(x)
        ax1.set_yticks(range(int(min(y_clean)) - 1, int(max(y_clean)) + 1, 1))
        ax1.grid(True, linestyle='--', axis='y')

        # 创建第二条折线（右边的y轴）
        ax2 = ax1.twinx()
        ax2.plot(x, y_attack, marker='^', linestyle='-', color='r', label='AUA%',linewidth=2)
        #ax2.set_ylabel('Accuracy Under Attack', fontsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_yticks(range(int(min(y_attack)) - 1, int(max(y_attack)) + 1, 1))
    else:
        ax1 = axs[i]
        ax1.plot(x, y_clean, marker='o', linestyle='-', color='b', label='Clean%')
        ax1.set_xlabel('r', fontsize=30)
        #ax1.set_ylabel('Clean Accuracy', fontsize=20)
        ax1.tick_params(axis='both', labelsize=20)
        ax1.set_xticks(x)
        ax1.set_yticks(range(int(min(y_clean)) - 1, int(max(y_clean)) + 1, 1))
        ax1.grid(True, linestyle='--', axis='y')

        # 创建第二条折线（右边的y轴）
        ax2 = ax1.twinx()
        ax2.plot(x, y_attack, marker='^', linestyle='-', color='r', label='AUA%',linewidth=2)
        #ax2.set_ylabel('Accuracy Under Attack', fontsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_yticks(range(int(min(y_attack)) - 1, int(max(y_attack)) + 1, 5))

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,fontsize=20)

    # 添加标题
    ax1.set_title(title, fontsize=30)

# 调整布局
plt.tight_layout()

# 保存图形为PDF文件
plt.savefig("r.pdf", format='pdf',dpi=500)

# 显示图形
plt.show()