import pandas as pd

# 读取原始的Excel文件
df = pd.read_excel('radult_results.xlsx')

# 将每个方法的结果展开成多列
df_expanded = df.explode('Results').reset_index(drop=True)

# 检查是否存在重复的方法名称
if df_expanded['Method'].duplicated().any():
    # 如果存在重复，我们需要对它们进行处理，以确保每个方法名称是唯一的
    # 这里我们简单地将重复的方法名称后加上一个序号来区分
    df_expanded['Method'] = df_expanded.groupby('Method')['Method'].cumcount().apply(lambda x: f"{df_expanded['Method']} ({x + 1})" if x > 0 else df_expanded['Method'])

# 将展开后的DataFrame转换为横向格式
horizontal_df = df_expanded.pivot(index='Method', columns='Results', values='Results')

# 由于Results列是字符串格式，我们将其转换为float类型
horizontal_df.columns = horizontal_df.columns.astype(float)

# 将横向格式的DataFrame保存到新的Excel文件
horizontal_df.to_excel('horizontal_results.xlsx')