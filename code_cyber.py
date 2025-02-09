# Import necessary libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD

df1 = pd.read_csv("C:/Users/schai/OneDrive/Desktop/Cybersecurity/Project/Wednesday-workingHours.pcap_ISCX.csv")

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

cols = df1.columns
cols = cols.map(lambda x: x.replace(' ', '_') )
df1['Label'] = df1['Label'].str.strip()
df1 = df1.drop(df1[df1['Label'] == 'DoS Hulk'].index)
print("class count: " , class_counts)
df1 = df1.drop(df1[df1['Label'] == 'DoS GoldenEye'].index)
df1 = df1.drop(df1[df1['Label'] == 'Heartbleed'].index)
temp_count = df1['Label'].value_counts()
print("temp_count: ", temp_count)
# Check the final shape of the DataFrame
print(f"After cleaning, the dataset contains {df1.shape[0]} instances and {df1.shape[1]} features.")
df1.Label[df1.Label=='BENIGN'] = 0
df1.Label[df1.Label =='DoS slowloris'] = 1
df1.Label[df1.Label =='DoS Slowhttptest'] = -1
df1["Label"].astype('Int64')

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
    plt.suptitle('Column Distribution in CICIDS2017 Dataset Wednesday DDoS', y=1.02)
    plt.savefig("ColumnDistribution_WednesdayDDoS.png", dpi=500)
    # plt.show()


print("***SEABORN***")

import matplotlib.pyplot as plt # plotting
plotPerColumnDistribution(df1, 20, 5)

import seaborn as sns
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
plt.savefig("Corr_Matrix_Wednesday_DDoS.png", dpi=500)
# plt.show()

# Select only numeric columns for correlation analysis
numeric_df = df1.select_dtypes(include=np.number)

# Compute correlation matrix
corr = numeric_df.corr()

# Visualize the correlation matrix using Seaborn
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Converting categorical variables to integers
cat_cols = df1.select_dtypes(include=['object', 'category']).columns
print(cat_cols)

cat_cols = df1.select_dtypes(include='object').columns
print(cat_cols)

print('Data type of each column of Dataframe :')
df1.info(verbose=True)

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

#Generating the Sequemces

# Function to generate sequences
# Define sequence length and overlap
seq_length = 10 # Sequence length
seq_overlap = 5 # Overlap between sequences

def generate_sequences(data):
    seqs = []
    for i in range(0, len(data) - seq_length + 1, seq_overlap):
        seqs.append(data.iloc[i:i+seq_length, :])
    return seqs

# Generate sequences for train, validation, and test sets
train_seqs = generate_sequences(train_data)
val_seqs = generate_sequences(val_data)
test_seqs = generate_sequences(test_data)

# Function to pad sequences
def pad_sequences(seqs):
    padded_seqs = []
    for seq in seqs:
        if len(seq) < seq_length:
            padded_seq = np.concatenate((seq, np.zeros((seq_length-len(seq), len(df1.columns)))), axis=0)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    return np.array(padded_seqs)

# Pad sequences for train, validation, and test sets
train_seqs = pad_sequences(train_seqs)
val_seqs = pad_sequences(val_seqs)
test_seqs = pad_sequences(test_seqs)

# Reshape input
train_X = np.reshape(train_seqs, (train_seqs.shape[0], train_seqs.shape[1], len(train_df.columns)))
val_X = np.reshape(val_seqs, (val_seqs.shape[0], val_seqs.shape[1], len(train_df.columns)))
test_X = np.reshape(test_seqs, (test_seqs.shape[0], test_seqs.shape[1], len(train_df.columns)))

print("-----Type of train X")
print(type(train_X))

# Reshape input
train_X = np.reshape(train_seqs, (train_seqs.shape[0], train_seqs.shape[1], train_seqs.shape[2]))
val_X = np.reshape(val_seqs, (val_seqs.shape[0], val_seqs.shape[1], val_seqs.shape[2]))
test_X = np.reshape(test_seqs, (test_seqs.shape[0], test_seqs.shape[1], test_seqs.shape[2]))

# Normalize data

train_X = (train_X - train_X.mean()) / train_X.std()
val_X = (val_X - train_X.mean()) / train_X.std()
test_X = (test_X - train_X.mean()) / train_X.std()

# Convert labels to categorical
# Convert labels to categorical
train_y = to_categorical(train_data["Label"].values[:train_X.shape[0]])
val_y = to_categorical(val_data["Label"].values[:val_X.shape[0]])
test_y = to_categorical(test_data["Label"].values[:test_X.shape[0]])

train_X = train_X.astype('float32')
train_y = train_y.astype('float32')
val_X = val_X.astype('float32')
val_y = val_y.astype('float32')

features=df1.iloc[:, :-1].values
labelfeature=df1.iloc[:,-1].values


#LSTM model 
print("*** LSTM Model ***")
# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
seq_length = 10 # set the sequence length
n_features = len(df1.columns) # number of features in the dataset

# Define model architecture
from keras.layers import Activation

# Define model architecture with ReLU activation function
def lstmmodel(test_X, test_y):

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(seq_length, n_features), return_sequences=True))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Activation('relu'))
    model.add(Dense(units=2, activation='softmax'))

    # Compile model with SGD optimizer
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), F1Score(name='f1_score')])


    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)
    print(model.summary())

    # Training LSTM model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stop])
    print(history)

    #accuracy plot
    print(history.history['accuracy'])
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['accuracy'],'o')
    plt.plot(history.history['val_accuracy'])
    # plt.plot(history.history['val_accuracy'],'o')
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy percentage')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='lower right')
    plt.show()

    test_X = test_X.astype('float32')  # Convert test_X to float32
    test_y = test_y.astype('float32')  # Convert test_y to float32 (if applicable)

    loss, accuracy, precision, recall, f1_score = model.evaluate(test_X, test_y)
    print("LSTM Test Loss:", loss)
    print("LSTM Test Accuracy:", accuracy)
    print("LSTM Test Precision:", precision)
    print("LSTM Test Recall:", recall)
    print("LSTM Test F1-Score:", f1_score)


# Define CNN model architecture
def cnnmodel(test_X, test_y):
    model_cnn = Sequential()

    # Add Convolutional layers
    model_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)))
    model_cnn.add(MaxPooling1D(pool_size=2))

    model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model_cnn.add(MaxPooling1D(pool_size=2))

    # Flatten layer
    model_cnn.add(Flatten())

    # Add fully connected layers
    model_cnn.add(Dense(64, activation='relu'))
    model_cnn.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model_cnn.add(Dense(2, activation='softmax'))  # Output layer with 2 neurons (for binary classification)

    # Compile CNN model
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), F1Score(name='f1_score')])


    # Print model summary
    print(model_cnn.summary())

    # Train the CNN model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history_cnn = model_cnn.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop])

    # Plot accuracy
    import matplotlib.pyplot as plt

    plt.plot(history_cnn.history['accuracy'])
    plt.plot(history_cnn.history['val_accuracy'])
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Percentage')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

    test_X = test_X.astype('float32')  # Convert test_X to float32
    test_y = test_y.astype('float32')  # Convert test_y to float32 (if applicable)

    loss, accuracy, precision, recall, f1_score = model_cnn.evaluate(test_X, test_y)
    print("CNN Test Loss:", loss)
    print("CNN Test Accuracy:", accuracy)
    print("CNN Test Precision:", precision)
    print("CNN Test Recall:", recall)
    print("CNN Test F1-Score:", f1_score)

lstmmodel(test_X, test_y)
cnnmodel(test_X,  test_y)
