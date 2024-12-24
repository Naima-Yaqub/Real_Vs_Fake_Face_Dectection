from Data_processing import train_ds, val_ds, test_ds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dotenv import load_dotenv
from tensorflow import keras
from model import model
import tensorflow as tf
from tensorflow.keras.models import load_model

load_dotenv()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
CHANNELS = int(os.getenv("CHANNELS"))
EPOCHS = int(os.getenv("EPOCHS"))

#history = None

#model.compile(
#    optimizer='adam',
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#    metrics=['accuracy']
#)

model.compile(optimizer='adam',
            #optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

#if os.path.exists("trained_model.h5"):
#    model = load_model("trained_model.h5")
#else:
#    print("No saved model found. Training a new model...")
#    history = model.fit(
#        train_ds,
#        epochs=EPOCHS,
#        batch_size=BATCH_SIZE,
#        validation_data=val_ds
#    )

history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)]
    )
folder_path = 'C:/Users/naima/VSCODE-FILES/ML Labs/ML_Project/Trained_Models'
model_path = os.path.join(folder_path, "trained_model.h5")
model.save("trained_model.keras")
print("Model trained and saved!")

#def model_fit():
#    history = model.fit(
#        train_ds,
#        epochs=EPOCHS,
#        batch_size=BATCH_SIZE,
#        validation_data=val_ds
#    )
#return history