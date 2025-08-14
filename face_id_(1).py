# to recursively delete data
# !rm -r /content/__MACOSX
# !rm -r /content/application_data
# !rm -r /kaggle/working/training_checkpoints

# !pip install tensorflow

# !pip install opencv-python

# unzipping
# !unzip -q /content/data.zip

import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,Layer,MaxPooling2D,Input,Flatten

# """**Data Collection**"""

pos_path = os.path.join('data','positive')
neg_path = os.path.join('data','negative')
anc_path = os.path.join('data','anchor')
len(os.listdir(pos_path))

# # os.makedirs(pos_path)
# # os.makedirs(neg_path)
# # os.makedirs(anc_path)

# # for directory in os.listdir('archive/lfw-deepfunneled') :
# #     person_path = os.path.join('archive/lfw-deepfunneled',directory)
# #     if os.path.isdir(person_path) :
# #         for image in os.listdir(person_path) :
# #             present_path = os.path.join(person_path,image)
# #             new_path = os.path.join(neg_path,image)
# #             os.replace(present_path,new_path)

# # cap = cv2.VideoCapture(1)
# # last_frame = None
# # while cap.isOpened() :
# #     ret,frame = cap.read()
# #     # Showing img on screen
# #     cv2.imshow('Live Feed',frame)

# #     if cv2.waitKey(1) & 0xFF == ord('a') :
# #         img_name = os.path.join(anc_path,f'{uuid.uuid1()}.jpg')
# #         cv2.imwrite(img_name,cv2.resize(frame,(250,250)))

# #     if cv2.waitKey(1) & 0xFF == ord('q') :
# #          img_name = os.path.join(pos_path,f'{uuid.uuid1()}.jpg')
# #          cv2.imwrite(img_name,cv2.resize(frame,(250,250)))
# #     # Breaking gracefully
# #     if cv2.waitKey(1) & 0xFF == ord('x') :
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# # Generators
# anchor = tf.data.Dataset.list_files(os.path.join(anc_path, '*.jpg')).take(300)
# positive = tf.data.Dataset.list_files(os.path.join(pos_path, '*.jpg')).take(300)
# negative = tf.data.Dataset.list_files(os.path.join(neg_path, '*.jpg')).take(300)

# dir_test = anchor.as_numpy_iterator()
# print(dir_test.next())

def preprocess(file_path) :
    byte_img = tf.io.read_file(file_path) # read image
    img = tf.io.decode_jpeg(byte_img) # load image
    img = tf.image.resize(img,(100,100)) # resize to 100*100 (used in Siamese network)
    img = img / 255.0 # normalize
    return img

# img = preprocess('data/anchor/000a3874-1f65-11f0-b239-021edee75f25.jpg')
# print(img.shape)
# print(plt.imshow(img)) # show Axes
# plt.imshow(img) # plot with matplotlib (shows image)
# plt.show()

# # Creating labelled Dataset
# positvies = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
# negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
# data = positvies.concatenate(negatives)

# samples = data.as_numpy_iterator()
# example = samples.next()


# # Preprocess this combined dataset
def preprocess_twin(input_image,validation_image,label) :
    return (preprocess(input_image),preprocess(validation_image),label)

# res = preprocess_twin(*example)
# plt.figure()
# plt.imshow(res[0])
# plt.title("Input Image")
# plt.axis('off')  # optional, to remove axis ticks

# plt.figure()
# plt.imshow(res[1])
# plt.title("Validation Image")
# plt.axis('off')  # optional, to remove axis ticks
# plt.show()
# print(f'Label : {res[2]}')

# # DataLoader Pipeline
# data = data.map(preprocess_twin)
# data = data.cache()
# data = data.shuffle(buffer_size=1024)



# """**Till now we have bundled anchor images with postive and negative iamges and added labels to them accordingly and then shuffled them**"""

# # training partition
# train_data = data.take(round(len(data) * 0.7))
# train_data = train_data.batch(16)
# train_data = train_data.prefetch(8) # starts preprocessing for the next batch

# # testing partition
# test_data = data.skip(round(len(data) * 0.7))
# test_data = test_data.take(round(len(data) * 0.3))
# test_data = test_data.batch(16)
# test_data = test_data.prefetch(8)

# # Defining our Embedding layer
def make_embedding() :
    inp = Input(shape = (100,100,3), name = 'input_image')

    # First block
    c1 = Conv2D(64,(10,10),activation = 'relu')(inp)
    m1 = MaxPooling2D(64,(2,2),padding = 'same')(c1)

    # Second block
    c2 = Conv2D(128,(7,7),activation = 'relu')(m1)
    m2 = MaxPooling2D(64,(2,2),padding = 'same')(c2)

    # Third block
    c3 = Conv2D(128,(4,4),activation = 'relu')(m2)
    m3 = MaxPooling2D(64,(2,2),padding = 'same')(c3)

    # Final Embedding block
    c4 = Conv2D(256,(4,4),activation = 'relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096,activation = 'sigmoid')(f1)

    return Model(inputs = [inp],outputs = [d1],name = 'embedding')

embeddings = make_embedding()
# embeddings.summary()

# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(*embeddings(input_image), *embeddings(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# siamese_network = siamese_model()
# # siamese_network.summary()



# # TRAINING Part
# binary_cross_entropy = tf.losses.BinaryCrossentropy()
# optimizer = tf.keras.optimizers.Adam(1e-4) # 0.0001
# # Establish checkpoints
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
# checkpoint = tf.train.Checkpoint(opt = optimizer,siamese_model = siamese_network)

# # Build train step fucntion
# @tf.function
# def train_step(batch) :
#     # Record all the operation
#     with tf.GradientTape() as tape :
#         x = batch[:2]
#         y = batch[2]
#         # Forward pass
#         yhat = siamese_network(x, training = True)
#         loss = binary_cross_entropy(y,yhat)
#     print(loss)
#     # Calculate gradients
#     grad = tape.gradient(loss,siamese_network.trainable_variables)
#     # Calculate updated weights and apply to siamese model
#     optimizer.apply_gradients(zip(grad,siamese_network.trainable_variables))
#     return loss

# # training loop
# def train(data,epochs) :
#     for epoch in range(1 , epochs + 1) :
#         print(f'\nEpoch {epoch} / {epochs}.')
#         progbar = tf.keras.utils.Progbar(len(data))
#         # looping through each batch
#         for idx,batch in enumerate(data) :
#             train_step(batch) # Running training step here
#             progbar.update(idx + 1)
#             # if epoch % 10 == 0 : # Saving to checkpoint
#             #     checkpoint.save(file_prefix = checkpoint_prefix)

# train(train_data,50)

# # Evaluate the model
# from tensorflow.keras.metrics import Recall,Precision

# test_input,test_val,y_true = test_data.as_numpy_iterator().next()

# test = test_data.as_numpy_iterator().next()
# len(test[1])

# # Make predictions
# yhat = siamese_network.predict([test_input,test_val])
# yhat

# def calculate_accuracy(yhat, y_true, threshold=0.5):
#     # Convert probabilities to binary predictions
#     y_pred = (yhat > threshold).astype(int)
#     # Compare with true labels
#     correct = (y_pred.flatten() == y_true).astype(int)
#     # Calculate accuracy
#     accuracy = correct.mean()
#     return accuracy

# acc = calculate_accuracy(yhat, y_true)
# print(f"Accuracy: {acc * 100:.2f}%")

# y_true

# # Processing post result
# [1 if prediction > 0.5 else 0 for prediction in yhat]

# # Creating metric objects
# r = Recall()
# p = Precision()

# # Calculating their value
# r.update_state(y_true,yhat)
# p.update_state(y_true,yhat)

# # Calling result
# print(r.result().numpy())
# print(p.result().numpy())

# # Visualize Results
# plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.imshow(test_input[2])
# plt.subplot(1,2,2)
# plt.imshow(test_val[2])
# plt.show()

# # Visualize Results
# plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.imshow(test_input[5])
# plt.subplot(1,2,2)
# plt.imshow(test_val[5])
# plt.show()

# # Save weights
# siamese_network.save('siamese_model.h5')

# # Reload model
model = tf.keras.models.load_model('siamesemodel2.h5',
                                   custom_objects = {'L1Dist':L1Dist,'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# # to calculate size of the folder
# # !du -sh /kaggle/working/siamese_model.h5


def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png'):
            input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

            # Make Predictions 
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.squeeze(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total
    verification = detection / len(results)
    verified = verification > verification_threshold
    return results, verified


# Real time verification
cap = cv2.VideoCapture(1)
while cap.isOpened() :
  ret,frame = cap.read()
  cv2.imshow('Verification',frame)
  # Verification trigger
  if cv2.waitKey(1) & 0xFF == ord('v') :
    # save image to input folder
    cv2.imwrite(os.path.join('application_data','input_image','input_image.jpg'),frame)
    results , verified = verify(model,0.9,0.7) # Run verification
    print(verified)
    # Breaking gracefully
  if cv2.waitKey(1) & 0xFF == ord('x') :
        break

cap.release()
cv2.destroyAllWindows()