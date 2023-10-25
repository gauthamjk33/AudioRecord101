# %%
import os
import librosa
import pandas as pd
import joblib

# %%
def load_audio_files(directory):
    audio_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
            audio_data.append((filename, audio, sr))  # Store the filename, audio data, and sample rate
    return audio_data

audio_directory = r'D:/release_in_the_wild'
audio_list = load_audio_files(audio_directory)

# %%
def extract_features(audio_data, n_mfcc=13):
    features = []
    for _, audio, sr in audio_data:  
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        features.append(mfccs)  # Append only the MFCCs, not the filename
    return features


audio_features = extract_features(audio_list, n_mfcc=13)

# %%
metadata_file = r'/home/gautham/release_in_the_wild/meta.csv'
metadata_df = pd.read_csv(metadata_file)
print(metadata_df.head(10))

# %%
metadata_df.columns = metadata_df.columns.str.strip()
merged_data = pd.merge(metadata_df, pd.DataFrame(audio_list, columns=['Filename', 'Audio', 'SampleRate']), left_on='file', right_on='Filename', how='inner')
print(merged_data.head(10))

# %%
from sklearn.model_selection import train_test_split

# Split the data into train (70%) and test (30%) sets
train_data, test_data = train_test_split(merged_data, test_size=0.3, random_state=42)

# Further split the test data into test (15%) and evaluation (15%) sets
test_data, eval_data = train_test_split(test_data, test_size=0.5, random_state=42)

# %%
# Convert MFCCs to a format suitable for machine learning
X = audio_features 
# = [1 if label == 'spoof' else 0 for label in labels]  # Convert labels to binary (1 for spoof, 0 for bona-fide)

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# Determine the maximum length of MFCC feature vectors
max_length = max(mfccs.shape[1] for mfccs in X)


def preprocess_mfccs(mfccs, max_length):
    if len(mfccs.shape) == 1:
        # Handle 1D MFCCs
        mfccs = np.expand_dims(mfccs, axis=0) 
    
    n_mfcc, n_frames = mfccs.shape
    
    if n_frames < max_length:
        # Pad with zeros if it's shorter than max_length
        padding = max_length - n_frames
        mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
    elif n_frames > max_length:
        # Truncate if it's longer than max_length
        mfccs = mfccs[:, :max_length]
    
    return mfccs

# Define the maximum length
max_length = 100  # You can adjust this value as needed

# Apply padding or truncation and reshape to all feature vectors
X_train = np.array([preprocess_mfccs(mfccs, max_length) for mfccs in train_data['Audio']])
X_test = np.array([preprocess_mfccs(mfccs, max_length) for mfccs in test_data['Audio']])
X_eval = np.array([preprocess_mfccs(mfccs, max_length) for mfccs in eval_data['Audio']])

# Reshape the data to 2D before applying StandardScaler
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_eval_reshaped = X_eval.reshape(X_eval.shape[0], -1)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)
X_eval_scaled = scaler.transform(X_eval_reshaped)

# %%
y_train = [1 if label == 'spoof' else 0 for label in train_data['label']]
y_test = [1 if label == 'spoof' else 0 for label in test_data['label']]
y_eval = [1 if label == 'spoof' else 0 for label in eval_data['label']]

# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train_scaled, y_train)

# Predict labels for the test and evaluation data
y_test_pred = rf_classifier.predict(X_test_scaled)
y_eval_pred = rf_classifier.predict(X_eval_scaled)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_test_pred)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)

print("Test Accuracy:", test_accuracy)
print("Evaluation Accuracy:", eval_accuracy)

# Generate a classification report for test data (includes precision, recall, F1-score, and more)
test_report = classification_report(y_test, y_test_pred)
print("Classification Report for Test Data:\n", test_report)

# %%
model_filename = r'in_the_wild.joblib'
joblib.dump(rf_classifier, model_filename)

print(f"Model saved as {model_filename}")

# Load the trained Random Forest classifier from the file
loaded_rf_classifier = joblib.load(model_filename)


# %%
def test_audio_file(audio_file_path, classifier, max_length):
    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Preprocess the MFCCs and ensure they have the correct shape
    mfccs_processed = preprocess_mfccs(mfccs, max_length)
    
    # Ensure that the MFCCs have the same number of features as max_length
    if mfccs_processed.shape[1] != max_length:
        raise ValueError(f"MFCC feature shape does not match max_length ({max_length}).")
    
    # Apply StandardScaler
    mfccs_scaled = scaler.transform(mfccs_processed) 
    
    # Predict using the loaded classifier
    prediction = classifier.predict(mfccs_scaled)
    
    # Convert the prediction to a human-readable label
    result = "spoof" if prediction[0] == 1 else "bona-fide"
    
    return result

# Test an audio file
test_audio_path = r'/home/gautham/release_in_the_wild/1000.wav'
result = test_audio_file(test_audio_path, loaded_rf_classifier, max_length=100)  
print(f"Audio file is {result}")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Define a simple CNN model using PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (max_length // 4) * (n_mfcc // 4), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * (max_length // 4) * (n_mfcc // 4))
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model
model = CNNModel()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_torch = torch.Tensor(X_train_scaled).unsqueeze(1)  # Add a channel dimension
y_train_torch = torch.Tensor(y_train).unsqueeze(1)
X_test_torch = torch.Tensor(X_test_scaled).unsqueeze(1)
y_test_torch = torch.Tensor(y_test).unsqueeze(1)

# Create DataLoader for training data
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
y_test_pred = (model(X_test_torch).detach().numpy() > 0.5).astype(int)
y_eval_pred = (model(torch.Tensor(X_eval_scaled).unsqueeze(1)).detach().numpy() > 0.5).astype(int)

# Convert predictions to numpy arrays
y_test_pred = y_test_pred.squeeze()
y_eval_pred = y_eval_pred.squeeze()

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
eval_accuracy = accuracy_score(y_eval, y_eval_pred)

print("Test Accuracy:", test_accuracy)
print("Evaluation Accuracy:", eval_accuracy)

# Generate a classification report for test data
test_report = classification_report(y_test, y_test_pred)
print("Classification Report for Test Data:\n", test_report)

# %%


# %%



