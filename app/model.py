import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# Set up data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Set up data generators without augmentation for validation and testing
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set batch size
batch_size = 32

# Load datasets using flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Ensure to keep the order for evaluation
)

# Build and compile your model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


steps_per_epoch = max(1, train_generator.samples // batch_size)
validation_steps = max(1, valid_generator.samples // batch_size)
test_steps = max(1, test_generator.samples // batch_size)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=66,
    validation_data=valid_generator,
    validation_steps=validation_steps
)


test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f'Test accuracy: {test_acc}')


test_generator.reset()  # Reset generator to start from beginning

predict_steps = len(test_generator)

# Predict the classes
predictions = model.predict(test_generator, steps=predict_steps)

# Convert predictions to binary classes
predicted_classes = (predictions > 0.5).astype("int32")


true_classes = test_generator.classes

predicted_classes = predicted_classes[:len(true_classes)]

# Compute confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_classes, predicted_classes, target_names=["benign", "malignant"])
print('Classification Report:')
print(class_report)

# Save the model
model.save('demo1.h5')

# Load the model
loaded_model = tf.keras.models.load_model('demo1.h5')

# Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('results/accuracy_plot.png')
plt.close()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('results/loss_plot.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Extract metrics from classification report
report_dict = classification_report(true_classes, predicted_classes, target_names=["benign", "malignant"], output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

# Visualize precision, recall, and f1-score
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_report.index[:-3], y=df_report[metric][:-3])  # Exclude support, accuracy, macro avg, and weighted avg
    plt.title(f'{metric.capitalize()} by Class')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Class')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'results/{metric}_plot.png')
    plt.close()