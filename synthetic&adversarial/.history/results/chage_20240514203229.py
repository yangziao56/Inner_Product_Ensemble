import pandas as pd

# 读取原始的Excel文件
df = pd.read_excel('adult_results.xlsx')

# 为了确保Method列中没有重复的条目，我们可以为每个方法生成一个唯一的标识符
df['Method_unique'] = df['Method'] + '_' + df.groupby('Method').cumcount().apply(lambda x: f"{x+1:02d}").astype(str)

# 将每个方法的结果展开成多列
df_expanded = df_exploded(df)

# 将展开后的DataFrame转换为横向格式
horizontal_df = df_expanded.pivot(index='Method_unique', columns='Results', values='Results')

# 由于Results列是字符串格式，我们将其转换为float类型
horizontal_df.columns = horizontal_df.columns.astype(float)

# 为了使结果更清晰，我们可以将'Method_unique'列中的标识符移除，恢复成原来的方法名称
horizontal_df.index = horizontal_df.index.str.split('_').str[0]

# 将横向格式的DataFrame保存到新的Excel文件
horizontal_df.to_excel('horizontal_results.xlsx')