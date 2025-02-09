import numpy as np
import pandas as pd
 
# df1 = pd.read_csv("\Users\akashsmac\Documents\Git\DDoS\DatasetWednesday.csv")
df1 = pd.read_csv("/Users/akashsmac/Documents/Git/DDoS/DatasetWednesday.csv")

print(df1.info())
print(f"The combined dataset contains {df1.shape[0]} instances and {df1.shape[1]} features.")
# Remove instances with missing class labels
df1 = df1[~df1['Label'].isnull()]
# Remove instances with missing information
df1 = df1.dropna()
# Check for duplicate rows
duplicates = df1.duplicated().sum()
print(f"The dataset1 contains {duplicates} duplicate rows.")
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.dropna(inplace=True)  
df1.drop_duplicates(inplace=True)
# Count the occurrences of each class label
class_counts = df1['Label'].value_counts()
print("Class label counts:")
print(class_counts)
print(df1.columns)
df1['Label'] = df1['Label'].str.strip()
df1 = df1.drop(df1[df1['Label'] == 'DoS Hulk'].index)
print("class count: " , class_counts)
df1 = df1.drop(df1[df1['Label'] == 'DoS GoldenEye'].index)
df1 = df1.drop(df1[df1['Label'] == 'Heartbleed'].index)
temp_count = df1['Label'].value_counts()
print("temp_count: ", temp_count)
# Check the final shape of the DataFrame
print(f"After cleaning, the dataset contains {df1.shape[0]} instances and {df1.shape[1]} features.")

#Visualisation of this dataset
import math
import matplotlib.pyplot as plt
import numpy as np

def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df1 if nunique[col] > 1 and nunique[col] < 80]] 
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = math.ceil((nCol + nGraphPerRow - 1) / nGraphPerRow)
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df1.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.suptitle('Column Distribution in CICIDS2017 Dataset Friday DDoS', y=1.02)
    plt.savefig("ColumnDistribution_FridayDDoS.png", dpi=500)
    # plt.show()


print("SEABORN")

import matplotlib.pyplot as plt # plotting
plotPerColumnDistribution(df1, 20, 5)

import seaborn as sns

# corr = df1.corr()

numeric_df = df1.select_dtypes(include=np.number)

# Compute correlation matrix
corr = numeric_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix Heatmap of CICIDS2017 Dataset Friday DDoS', fontsize=20)
plt.tight_layout()  # Adjusts the padding
plt.savefig("Corr_Matrix_Friday_DDoS.png", dpi=500)
# plt.show()

# Select only numeric columns for correlation analysis
# numeric_df = df1.select_dtypes(include=np.number)

# # Compute correlation matrix
# corr = numeric_df.corr()

# # Visualize the correlation matrix using Seaborn
# import seaborn as sns
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

#Training the dataset
# Shuffle the rows
df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)

# Select a subset of the data for training
num_data = 15000 # can change between 500 and 1152382
train_df = df1.iloc[:num_data].copy() # Use .iloc to avoid a SettingWithCopyWarning

print('Data type of each column of Dataframe :')
train_df.info(verbose=True)

# Splitting the data

train_size = int(0.7 * len(train_df))
val_size = int(0.15 * len(train_df))
test_size = len(train_df) - train_size - val_size
train_data = train_df.iloc[:train_size, :]
val_data = train_df.iloc[train_size:train_size+val_size, :]
test_data = train_df.iloc[train_size+val_size:, :]