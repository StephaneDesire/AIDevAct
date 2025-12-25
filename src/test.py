import pandas as pd

# Test lecture pull_requests
df = pd.read_parquet("/home/stephane/Documents/AIDevAct/data/intermediate/pr_human_filtered.parquet")
print(df.head(100)['agent'])
print(df.columns)
print(len(df))

"""df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
print(df.head())
print(df.columns)
print(len(df))

df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")
print(df.head())
print(df.columns)
print(len(df))
"""