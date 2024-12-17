import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "C:/Users/DG-CPRI/Desktop/dataset/train"
valid_dir = "C:/Users/DG-CPRI/Desktop/dataset/test"

# Image Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            
    rotation_range=20,         
    width_shift_range=0.2,     
    height_shift_range=0.2,    
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    fill_mode='nearest'        
)

valid_datagen = ImageDataGenerator(rescale=1./255) 

# Loading and preprocessing the data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

base_model.trainable = False

# Building the model
model = models.Sequential([
    base_model,  
    layers.GlobalAveragePooling2D(),  
    layers.Dense(1024, activation='relu'), 
    layers.Dropout(0.5),  
    layers.Dense(len(train_generator.class_indices), activation='softmax')  
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('skin_disease_model.keras', save_best_only=True)
    ]
)

model.save('skin_disease_model.keras')
 