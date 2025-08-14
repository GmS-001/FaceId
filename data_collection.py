import cv2
import os
import random
import numpy
import uuid # Universal unique identifier
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Layer, MaxPooling2D, Input, Flatten
from tensorflow.keras.models import Model

# pos_path = os.path.join('data','positive')
# neg_path = os.path.join('data','negative')
# anc_path = os.path.join('data','anchor')
# os.makedirs(pos_path)
# os.makedirs(neg_path)
# os.makedirs(anc_path)

# for directory in os.listdir('archive/lfw-deepfunneled') :
#     person_path = os.path.join('archive/lfw-deepfunneled',directory)
#     if os.path.isdir(person_path) : 
#         for image in os.listdir(person_path) :
#             present_path = os.path.join(person_path,image)
#             new_path = os.path.join(neg_path,image)
#             os.replace(present_path,new_path)


# cap = cv2.VideoCapture(1)
# last_frame = None
# while cap.isOpened() :
#     ret,frame = cap.read()
#     frame = frame[350:-50,700:-600,:]
#     # Showing img on screen
#     cv2.imshow('Live Feed',frame)

#     if cv2.waitKey(1) & 0xFF == ord('a') :
#         img_name = os.path.join(anc_path,f'{uuid.uuid1()}.jpg')
#         cv2.imwrite(img_name,cv2.resize(frame,(250,250)))

#     if cv2.waitKey(1) & 0xFF == ord('q') :
#          img_name = os.path.join(pos_path,f'{uuid.uuid1()}.jpg')
#          cv2.imwrite(img_name,cv2.resize(frame,(250,250)))
#     # Breaking gracefully
#     if cv2.waitKey(1) & 0xFF == ord('x') :
#         break 

# cap.release()
# cv2.destroyAllWindows()

print(len(os.listdir(os.path.join('application_data','verification_images'))))