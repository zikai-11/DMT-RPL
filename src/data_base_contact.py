import pandas as pd

# 读取所有CSV文件
df1 = pd.read_csv('attack_base_data/ours/yelp/textfooler-ours-500.csv')
df2 = pd.read_csv('attack_base_data/ours/yelp/pwws-ours-500.csv')
df3 = pd.read_csv('attack_base_data/ours/yelp/bertattack-ours-500.csv')
df4 = pd.read_csv('attack_base_data/ours/yelp/deepwordbug-ours-500.csv')

# 纵向合并（追加行）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存合并后的文件
combined_df.to_csv('attack_base_data/ours/yelp/combined_file.csv', index=False)