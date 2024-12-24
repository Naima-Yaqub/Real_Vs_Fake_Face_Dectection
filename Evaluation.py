#if __name__ == "__main__":
import tensorflow as tf
from Training import history
from model import model 
from Data_processing import train_ds, val_ds, test_ds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt 
from Data_processing import class_name
from dotenv import load_dotenv
import numpy as np 

load_dotenv()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
CHANNELS = int(os.getenv("CHANNELS"))
EPOCHS = int(os.getenv("EPOCHS"))

print(history)
print(history.params)
print(history.history.keys())

train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {val_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Training Loss: {train_loss:.2f}")
print(f"Validation Loss: {val_loss:.2f}")
print(f"Test loss: {test_loss:.2f}")

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']

loss= history.history['loss']
val_loss = history.history['val_loss']


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

epochs_range = range(len(acc))  

plt.figure(figsize=(8, 8))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()


for images_batch, labels_batch in test_ds.take(1):

    # Getting the image and label
    second_image = images_batch[5].numpy().astype('uint8')
    second_label = labels_batch[5].numpy()

    # Getting predictions for the batch
    batch_prediction = model.predict(images_batch)

    # Threshold 
    threshold = 0.5

    # Converting predictions to 'real' or 'fake' based on the threshold
    batch_predictions_labels = ["real" if pred >= threshold else "fake" for pred in batch_prediction.flatten()]

    # Getting the predicted label for the second image
    second_image_prediction = batch_predictions_labels[5]
    second_image_confidence = batch_prediction[5][0]  # Probability of the second image

    # Ploting the second image
    plt.imshow(second_image)
    
    # Add the actual label, predicted label, and confidence on top of the image
    plt.title(f"Actual: {class_name[second_label]} | Pred: {second_image_prediction} | Confidence: {second_image_confidence:.2f}")
    plt.axis('off')  

    plt.show()

    print(f'Actual label: {class_name[second_label]}')
    print(f'Predicted label: {second_image_prediction}')
    if class_name[second_label]== 'Real':
        print(f'Confidence: {second_image_confidence:.2f}')
    else:
        print(f'Confidence: {1-second_image_confidence:.2f}')  
    

def predict_and_display(model, test_ds, class_name, image_index):
    # Get a batch of images and labels
    for images_batch, labels_batch in test_ds.take(1):

        # Getting the specific image and label
        image = images_batch[image_index].numpy().astype('uint8')
        label = labels_batch[image_index].numpy()

        # Geting predictions for the batch
        batch_prediction = model.predict(images_batch)

        #  Threshold
        threshold = 0.5

        # Converting predictions to 'real' or 'fake' based on the threshold
        batch_predictions_labels = ["real" if pred >= threshold else "fake" for pred in batch_prediction.flatten()]

        # Get the predicted label and confidence for the selected image
        image_prediction = batch_predictions_labels[image_index]
        image_confidence = batch_prediction[image_index][0]  # Probability of the image

        # Plot the image
        plt.imshow(image)
        
        # Add the actual label, predicted label, and confidence on top of the image
        plt.title(f"Actual: {class_name[label]} | Pred: {image_prediction}\nConfidence: {image_confidence:.2f}")
        plt.axis('off') 


plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Call the predict_and_display function for each image
        predict_and_display(model, test_ds, class_name, image_index=i)

plt.show()


#for images_batch, labels_batch in test_ds.take(1):
#    first_image = images_batch[0].numpy().astype('uint8')
#    first_label = labels_batch[0].numpy()

#    print('first image to predict')
#    plt.imshow(first_image)
#    print('actual label: ', class_name[first_label])

#    batch_prediction= model.predict(images_batch)
#    print('Predicted Label: ',class_name[np.argmax(batch_prediction[0])])

#def predict(model,img):
#    #img_array = tf.keras.preprocessing.image.img_to_array(img) #[images[i].numpy()]
#    img_array = tf.expand_dims(img,0)

#    predictions = model.predict(img_array)

#    predict_class= class_name[np.argmax(predictions[0])]
#    confidence = round(100 * (np.max(predictions[0])),2)
#    return predict_class, confidence

#for images, labels in test_ds.take(1):
#    for i in range(9):
#      ax= plt.subplot (3,3,i+1)
#      plt.imshow(images[i].numpy().astype('uint8'))

#      predict_class, confidence = predict(model, images[i].numpy())
#      actual_class = class_name[labels[i]]
#      plt.title(f"Actual: {actual_class}\nPredicted: {predict_class} ({confidence}%)")

#      plt.axis('off')
#plt.show()
#print('Prediction done!')

