import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('database1.csv', encoding='utf-8')

# 2. 数据预处理
# 性别：A=1, B=2
df['性别'] = df['1.您的性别'].map({'A.男': 1, 'B.女': 2})

# 年级：A=1, B=2, C=3, D=4, E=5
df['年级'] = df['3.您的年级'].map({'A.大一': 1, 'B.大二': 2, 'C.大三': 3, 'D.大四': 4, 'E.大五及以上': 5})

# 是否参加/了解社会实践：A=1, B=0
df['是否参加社会实践'] = df['4.您是否参加/了解预防医学专业社会实践的主要形式？'].map({'A.是': 1, 'B.否': 0})

social_practice_types = df['5.若是，参加过以下哪些？'].str.split('┋').apply(pd.Series)
for col in social_practice_types.columns:
    df[f'社会实践类型_{col+1}'] = social_practice_types[col].map(lambda x: 1 if pd.notnull(x) else 0)

df = df.replace(r'^\s*$', np.nan, regex=True)

# 确保所有数值列都是数值型
numeric_cols = df.columns[7:16]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# 3. 从年级角度分析社会实践各项知识（H列到P列）
grade_stats = df.groupby('年级')[numeric_cols].agg(['mean', 'median', 'std'])
for col in numeric_cols:
    print(f"\n{col}的差异性分析：")
    for i in range(1, 6):
        for j in range(i+1, 6):
            group1 = df[(df['年级'] == i) & (df[col].notna())][col]
            group2 = df[(df['年级'] == j) & (df[col].notna())][col]
            if len(group1) >= 2 and len(group2) >= 2:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                print(f"年级{i}和年级{j}在{col}上的t检验结果：t统计量={t_stat:.2f}, p值={p_value:.4f}")

# 4. 输出结果
print("\n各年级社会实践知识常规统计：")
for col in numeric_cols:
    print(f"\n{col}的统计结果：")
    print(grade_stats[col])
    # 可视化分析
# 各年级在每列上的分布
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    data = [df[df['年级'] == i][col].dropna() for i in range(1, 6)]
    plt.boxplot(data, labels=['大一', '大二', '大三', '大四', '大五及以上'])
    plt.title(f'各年级在{col}上的分布')
    plt.xlabel('年级')
    plt.ylabel('值')
    plt.show()
