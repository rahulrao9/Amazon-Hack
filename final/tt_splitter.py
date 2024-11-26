import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('balanced_output_1000.csv')

# Split the dataset into 80% training and 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets to CSV files
train_df.to_csv('B_train_800.csv', index=False)
test_df.to_csv('B_test_200.csv', index=False)