from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_FILE = Path(r"\\raoariaapps\raoariaapps$\Utilities\tb_adaption\registration_log.csv")
LOG_FILE_ANALYSIS = Path(r"\\raoariaapps\raoariaapps$\Utilities\tb_adaption\registration_log_analysis.csv")


df = pd.read_csv(LOG_FILE)

# translations
df['it_x'] = df['initial_transform'].str.split(",").str.get(0).astype(float)
df['it_y'] = df['initial_transform'].str.split(",").str.get(1).astype(float)
df['it_z'] = df['initial_transform'].str.split(",").str.get(2).astype(float)
df['tt_x'] = df['fine_tuned_transform'].str.split(",").str.get(0).astype(float)
df['tt_y'] = df['fine_tuned_transform'].str.split(",").str.get(1).astype(float)
df['tt_z'] = df['fine_tuned_transform'].str.split(",").str.get(2).astype(float)
df['ft_x'] = df['final_transform'].str.split(",").str.get(0).astype(float)
df['ft_y'] = df['final_transform'].str.split(",").str.get(1).astype(float)
df['ft_z'] = df['final_transform'].str.split(",").str.get(2).astype(float)

# shifts
df['tt-it_x'] = df['tt_x'] - df['it_x']
df['tt-it_y'] = df['tt_y'] - df['it_y']
df['tt-it_z'] = df['tt_z'] - df['it_z']
df['ft-it_x'] = df['ft_x'] - df['it_x']
df['ft-it_y'] = df['ft_y'] - df['it_y']
df['ft-it_z'] = df['ft_z'] - df['it_z']


df_s = df[df['accepted'] == True]
df.to_csv(LOG_FILE_ANALYSIS, index=False)



print("Shifts from initial to tuned")
for col in ['x', 'y', 'z']:
    print(f"{col}: {df_s[f'tt-it_{col}'].abs().max()}")

print()
print("Shifts from initial to final")
for col in ['x', 'y', 'z']:
    print(f"{col}: {df_s[f'ft-it_{col}'].abs().max()}")

print("Normalized mutual information")
print(df_s["normalized_mutual_information"].describe())
plt.hist(df_s["normalized_mutual_information"])
plt.show()
