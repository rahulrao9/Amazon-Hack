import pandas as pd
import ast

# Load your dataset
df = pd.read_csv('output_1000.csv')

# Convert string representations of lists in 'output' column to actual lists
df['output'] = df['output'].apply(ast.literal_eval)

# Get the unique counts for the specified combinations
count_0010 = len(df[df['output'].apply(lambda x: x == [0, 0, 1, 0])])
count_0100 = len(df[df['output'].apply(lambda x: x == [0, 1, 0, 0])])
count_1000 = len(df[df['output'].apply(lambda x: x == [1, 0, 0, 0])])

# Compute the mean of the counts
mean_count = int((count_0010 + count_0100 + count_1000) / 3)

# Get rows where the output is [0, 0, 0, 1]
df_0001 = df[df['output'].apply(lambda x: x == [0, 0, 0, 1])]

# Randomly sample mean_count rows from [0, 0, 0, 1]
df_0001_sampled = df_0001.sample(n=mean_count, random_state=42)

# Keep all rows with outputs not equal to [0, 0, 0, 1]
df_balanced = df[df['output'].apply(lambda x: x != [0, 0, 0, 1])]

# Concatenate the sampled rows from [0, 0, 0, 1] back to the dataset
df_final = pd.concat([df_balanced, df_0001_sampled])

# Save the balanced dataset to a new CSV file
df_final.to_csv('balanced_output_1000.csv', index=False)
