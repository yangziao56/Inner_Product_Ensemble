import pandas as pd

# 读取原始的Excel文件
df = pd.read_excel('adult_results.xlsx')

# 将每个方法的结果展开成多列
df_expanded = df.explode('Results').reset_index(drop=True)

# 为了避免pivot时出现重复项错误，这里我们不直接修改Method列，而是在pivot后进行处理

# 将展开后的DataFrame转换为横向格式
horizontal_df = df_expanded.pivot(index='Method', columns='Results', values='Results')

# 由于横向的列名是浮点数，我们需要将它们转换为字符串以保持数据的一致性
horizontal_df.columns = horizontal_df.columns.astype(str)

# 重置索引，将原来的'Method'变成列的一部分
horizontal_df = horizontal_df.reset_index()

# 检查索引（即原始的'Method'列）是否有重复，如果有，添加后缀以区分
duplicates = horizontal_df['Method'].duplicated(keep=False)
if any(duplicates):
    suffixes = [''] + ['_' + str(i+1) for i in range(sum(duplicates))]
    horizontal_df['Method'] = horizontal_df.apply(lambda row: row['Method'] + suffixes[duplicates[row.name]], axis=1)

# 由于我们添加了后缀，现在需要重新设置列标题
new_columns = [f"{method}{suffix}" if suffix else method for method, suffix in zip(horizontal_df['Method'], horizontal_df.columns.drop('Method'))]

# 使用新列标题更新DataFrame
horizontal_df[new_columns] = horizontal_df.drop(columns=['Method', 'Results'])

# 保存到新的Excel文件
horizontal_df.to_excel('horizontal_results.xlsx', index=False)