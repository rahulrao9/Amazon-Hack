import pandas as pd
import ast
from datasets import load_dataset

def process_csv(input_file, output_file):
    # Read the CSV file
    # dataframe = load_dataset("Hemabhushan/AmazonMLChallengeStage1", "P1")
    # df = dataframe["train"].to_pandas()
    df = pd.read_csv(input_file)
    result = []
    for _, row in df.iterrows():
        # Parse the complex list structures
        embeddings_list = ast.literal_eval(row['embeddings_list'])
        # print(type(embeddings_list))
        text_list = ast.literal_eval(row['text_list'])
        z_bar_list = ast.literal_eval(row['z_bar_list'])

        if len(text_list) > 0:
            for i in range(len(text_list)):
                parse_value = text_list[i]
                if parse_value == row['depth']:
                    output = [1, 0, 0, 0]
                elif parse_value == row['width']:
                    output = [0, 1, 0, 0]
                elif parse_value == row['height']:
                    output = [0, 0, 1, 0]
                else:
                    output = [0, 0, 0, 1]

                result.append({
                    'image_link': row['image_link'],
                    'group_id': row['group_id'],
                    'HEIGHT': row['height'],
                    'WIDTH': row['width'],
                    'DEPTH': row['depth'],
                    'EMBEDDING': embeddings_list[i][0] if i < len(embeddings_list) else None,
                    'PARSE_VALUE': parse_value,
                    'z_bar': z_bar_list[i] if i < len(z_bar_list) else None,
                    'output': output
                })

    # Create a new dataframe and save to CSV
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_file, index=False)

# Usage
input_file = 'final_output_1000.csv'  # Replace with your input file name
output_file = 'output_1000.csv'  # Replace with your desired output file name
process_csv(input_file, output_file)