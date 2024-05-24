import pandas as pd

# 加载Excel文件
file_path = 'nlp_results.xlsx'
xls = pd.ExcelFile(file_path)

# 读取第一个工作表
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 为每个方法的结果添加编号
df['result_number'] = df.groupby('Method').cumcount() + 1

# 透视表转换
df_pivot = df.pivot(index=None, columns=['Method', 'result_number'], values='Results')

# 扁平化多级列索引并格式化
df_pivot.columns = [f"{method}_{num}" for method, num in df_pivot.columns]
df_pivot.reset_index(drop=True, inplace=True)

# 保存转换后的DataFrame到新的Excel文件
output_file_path = 'nlp_results_transformed.xlsx'
df_pivot.to_excel(output_file_path, index=False)
