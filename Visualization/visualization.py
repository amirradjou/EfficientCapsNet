import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('train_log.csv')

# Extract the training and validation loss and accuracy values from the DataFrame
train_loss = df['loss'].tolist()
train_acc = df['accuracy'].tolist()
val_loss = df['val_loss'].tolist()
val_acc = df['val_accuracy'].tolist()

# Extract the epoch numbers from the DataFrame
epochs = df['epoch'].tolist()

# Plot the training loss and accuracy values against the epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, train_loss, label='Training loss')
ax1.plot(epochs, val_loss, label='Validation loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, train_acc, label='Training accuracy')
ax2.plot(epochs, val_acc, label='Validation accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Show the plot
plt.show()

