import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(2, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(18, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(AveragePooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_data.flow_from_directory(
    'imagenet_train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_data.flow_from_directory(
    'imagenet_val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)



score = model.evaluate_generator(val_generator, steps=len(val_generator))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
