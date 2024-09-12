import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CSV data
data = pd.read_csv('data/fer2013.csv')

# Separate features (pixels) and labels (emotion)
pixels = data['pixels']
emotions = data['emotion']

# Convert pixels to numpy array and reshape
X = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
X = X.reshape(-1, 48, 48, 1)  # assuming images are 48x48 grayscale

# Convert emotions to categorical labels
Y = to_categorical(emotions)

# Shuffle and split data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.22))

emotion_model.add(Flatten())
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(256, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
emotion_model_info = emotion_model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=40,
    validation_data=(X_val, Y_val),
    verbose=2
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(emotion_model_info.history['accuracy'], label='Accuracy')
plt.plot(emotion_model_info.history['val_accuracy'], label='Validation Accuracy')
plt.plot(emotion_model_info.history['loss'], label='Loss')
plt.plot(emotion_model_info.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training History')
plt.show()

# Make predictions on validation data
Y_pred = emotion_model.predict(X_val)

# Get predicted labels
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Get true labels
Y_true = np.argmax(Y_val, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_true, Y_pred_classes)

# Display confusion matrix
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=emotion_dict.values())
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Save the model
emotion_model.save('model/emotion_model.h5')