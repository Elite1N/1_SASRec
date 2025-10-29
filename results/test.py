import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re # for regular expressions
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='beauty_test1', help='Name of the results file')
args = parser.parse_args()
data = []
# Read the test file and parse.
with open(f'./results/{args.input_file}.txt', 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  # Skip header line
        parts = line.strip().split()
        #print (parts)      
        
        epoch = int(parts[0])
        
        # Extract float values using regex 
        numbers = re.findall(r'\d+\.\d+', line) # Matches any float numbers (no np.float64 prefix)
        #print (numbers)
        
        val_ndcg = float(numbers[0])
        val_hr = float(numbers[1])
        test_ndcg = float(numbers[2])
        test_hr = float(numbers[3])
        
        
        data.append({
            'epoch': epoch,
            'val_ndcg': val_ndcg,
            'val_hr': val_hr,
            'test_ndcg': test_ndcg,
            'test_hr': test_hr
        })
df = pd.DataFrame(data)
#print(df)

# Unpivot: combine all metrics into the same column
df_melted = df.melt(id_vars=['epoch'], 
                     var_name='metric', 
                     value_name='score')

#print (df_melted.head())
# TO DO: Seperate into ndcg and hr plots
# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted[df_melted['metric'].str.contains('ndcg')], x='epoch', y='score', hue='metric', palette = ['gray', 'red'], marker='o')
plt.title('Model Performance over Epochs (NDCG)')
plt.xlabel('Epoch')
plt.ylabel('NDCG@10')
handles, labels = plt.gca().get_legend_handles_labels() # It just works
plt.legend(title='',  loc='lower right',handles = handles, labels=['Validation Set', 'Testing Set'])
plt.grid(True)
plt.savefig(f'./results/{args.input_file}_performance_ndcg.png')

# Plotting for HR
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted[df_melted['metric'].str.contains('hr')], x='epoch', y='score', hue='metric', palette = ['gray', 'red'], marker='o')
plt.title('Model Performance over Epochs (HR)')
plt.xlabel('Epoch')
plt.ylabel('HR@10')
handles, labels = plt.gca().get_legend_handles_labels() # It just works
plt.legend(title='',  loc='lower right',handles = handles, labels=['Validation Set', 'Testing Set'])
plt.grid(True)
plt.savefig(f'./results/{args.input_file}_performance_hr.png')
plt.show()