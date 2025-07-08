import pandas as pd

# 读取两组数据
df1 = pd.read_csv('800wCV.TXT', delimiter=',', header=None)  # 假设数据没有标题行
df2 = pd.read_csv('svmyuce.txt', delimiter=',', header=None)  # 假设数据没有标题行

# 确定序列号和标签的列索引（假设是倒数第二列和最后一列）
sequence_col_index = -2
label_col_index = -1

# 创建映射字典，将第二批数据的序列号映射到标签
sequence_label_map = pd.Series(df2.iloc[:, label_col_index], index=df2.iloc[:, sequence_col_index]).to_dict()

# 使用映射字典更新第一组数据的标签列
df1.iloc[:, label_col_index] = df1.iloc[:, sequence_col_index].map(sequence_label_map)

# 检查更新后的数据
print(df1.head())

# 将更新后的数据保存到新的文本文件
df1.to_csv('svm-result.txt', sep=',', index=False, header=False)