import pandas as pd

# 假设df是原始的DataFrame
df = pd.read_excel('adult_results.xlsx')

# 将每个方法的结果展开成多列
df_expanded = df.explode('Results').reset_index(drop=True)

# 将展开后的DataFrame转换为横向格式
horizontal_df = df_expanded.pivot(index='Method', columns='Results', values='Results')

# 将Results列的名称转换为数值格式
horizontal_df.columns = horizontal_df.columns.astype(float)

# 将DataFrame转换为横向格式后保存到Excel文件
horizontal_df.to_excel('adult_results_new.xlsx')