import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split

resources_path = 'resources/'  # Base path for videos
target_size = (100, 100)  # Resize frames to this size
input_shape = (target_size[0], target_size[1], 3)

def augment_image(image):
    augmented_images = []

    augmented_images.append(image)

    for angle in [0, 90, 180, 270]:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (width, height))
        augmented_images.append(rotated_image)

    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)

    return augmented_images

def load_data(resources_path):
    data = []
    labels_list = []
    labels = os.listdir(resources_path)
    
    for label in labels:
        label_path = os.path.join(resources_path, label)
        if os.path.isdir(label_path):
            img_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
            for img_file in img_files:
                img_full_path = os.path.join(label_path, img_file)
                frame = cv2.imread(img_full_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                augmented_images = augment_image(frame)
                data.extend(augmented_images)
                labels_list.extend([int(label)] * len(augmented_images))
            
    return np.array(data), np.array(labels_list)

X, y = load_data(resources_path)
X = X.astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)
X[0].shape
# plt.imshow(X[0])
# plt.show()

def create_cnn(input_shape, num_classes):
    weight_decay = 0.3
    model = models.Sequential()
    model.add(layers.Conv2D(32, (10, 10), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(126, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay)))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

model = create_cnn(input_shape, len(np.unique(y)))
model.summary()

model_checkpoint = callbacks.ModelCheckpoint(
    'models', 
    save_weights_only=False, 
    save_best_only=True,
    verbose=1, 
    monitor='val_loss'
)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    shuffle=True,
                    callbacks=[model_checkpoint, early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()