import pandas as pd

# Caricare il dataset
df = pd.read_csv('datasets/dataset_fifa_15_23.csv')

print(df['overall'].describe())
