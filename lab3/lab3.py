import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
import json 
import random as r
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Dense, Activation
from sklearn.metrics import roc_auc_score

import seaborn as sns


def train_test_split(data,test_size):
    """test size in persentage"""
    X_train = []
    X_test = []
    r.shuffle(data)
    for img in data:
        if len(X_test)/len(data) < test_size:
            X_test.append(img)
        else:
            X_train.append(img)
    return X_train, X_test


def get_key_pts(img):
    algorithm = cv2.SIFT_create().detectAndCompute
    # gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = img
    # gray*=255
    # gray=gray.astype('uint8')
    _, descr = algorithm(gray,None)
    return descr.flatten().astype('float32')

def to_same_dims(array, dim):
  kp = np.zeros([len(array),dim])
  for i,j in enumerate(array):
      if len(j)>dim:
        kp[i][0:dim] = j[0:dim]
      else:
        kp[i][0:len(j)] = j
  return kp

def load_desc_model(height, num_classes):
  
  model = Sequential()
  model.add(Dense(64,input_shape=(height,)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(Dense(64))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Dense(128))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(128))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Dense(256))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
 
	# softmax classifier
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))

  # initiate Adam optimizer
  opt = Adam(learning_rate=1e-3, decay=1e-3/32)

  # Let's train the model using Adam
  model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
                #tf.keras.metrics.CategoricalAccuracy(),
                
  
  return model



if __name__ == "__main__":
    # other_images = os.listdir('lab2/img1')
    # folder_name = 'data'
    # image_for_train = os.listdir(f'lab2/{folder_name}')

    # random_data = np.random.choice(other_images,size=70)
    # print("here all is ok")
    # data = []
    # for img in random_data:
    #     data.append(img)
    # data += image_for_train
    # X_train, X_test = train_test_split(data,0.5)
    # y_train = []
    # y_test = []
    # for img in X_train:
    #     if "I" == img[0]:
    #         y_train.append(0)
    #     else: 
    #         y_train.append(1)
    # for img in X_test:
    #     if "I" == img[0]:
    #         y_test.append(0)
    #     else: 
    #         y_test.append(1)
    
    # print("here and all is ok 2")
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    # y_train = np.array(y_train)

    # x_test_desc = []
    # for img in X_test:
    #     if "I" == img[0] or "p" == img[0]:
    #         des = get_key_pts(cv2.imread("lab2/img1/" + str(img),0))
    #     else:
    #         des = get_key_pts(cv2.imread("lab2/data/" + str(img),0))
    #     des/=255.0
    #     x_test_desc.append(des)
    # print('here and all is ok 3')
    # x_test_desc = to_same_dims(x_test_desc, 256)
    # x_train_desc = []
    # for img in X_train:
    #     if "I" == img[0] or "p" == img[0]:
    #         des = get_key_pts(cv2.imread("lab2/img1/" + str(img),0))
    #     else:
    #         des = get_key_pts(cv2.imread("lab2/data/" + str(img),0))
    #     des/=255.0
    #     x_train_desc.append(des)
    # x_train_desc = to_same_dims(x_train_desc, 256)
    # print('here and all is ok 4')

    # np.savetxt('x_train.csv',x_train_desc, delimiter=',')
    # np.savetxt('x_test.csv',x_test_desc, delimiter=',')
    # np.savetxt('y_test.csv',y_test, delimiter=',')
    # np.savetxt('y_train.csv',y_train, delimiter=',')
    x_train_desc = np.genfromtxt('x_train.csv',delimiter=',')
    x_test_desc = np.genfromtxt('x_test.csv', delimiter=',')
    y_train = np.genfromtxt('y_train.csv',delimiter=',')
    y_test = np.genfromtxt('y_test.csv', delimiter=',')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                            factor=0.2,
                                                            patience=5, 
                                                            min_lr=1e-5)
    height = 256
    num_classes = 2
    model_desc = load_desc_model(height=height,
                                num_classes=num_classes)
    model_desc.summary()

    H_des = model_desc.fit(x_train_desc, y_train,  
                            batch_size=16,   
                            epochs=20, 
                            class_weight=None,
                            #   steps_per_epoch=None,
                            validation_data=(x_test_desc, y_test),
                            callbacks=[reduce_lr],verbose=1)
    batch_size = 32

    y_pred = model_desc.predict(x_test_desc,batch_size=batch_size)
    print(classification_report(y_test, np.argmax(y_pred,axis=-1)))
    # print(classification_report(y_test,y_pred))
    # print(roc_auc_score(y_test, y_pred))
    from sklearn.metrics import confusion_matrix, classification_report

    cf_matrix = confusion_matrix(y_test, [round(i[1]) for i in y_pred])
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    plt.show()
    cap = cv2.VideoCapture("lab3/video_2020-11-01_13-52-43.mp4")

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('test_video.avi', cv2.VideoWriter_fourcc(*'DIVX'),60,(width, height), isColor=True)

    while(cap.isOpened()):

        ret, frame = cap.read()
    
        data = get_key_pts(frame)
        data /= 255.0
        data = to_same_dims([list(data)],256)
        pred = model_desc.predict(data,verbose=0)
        text_to_output = 'Class ' + str(round(pred[0][1]))
        # cv2.imshow('webcam(2)',frame)
        out.write(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        if ret:
            cv2.putText(frame, text_to_output, org = (50,50),  
                    fontScale = 1, color = (255, 0, 0), thickness = 2, fontFace = cv2.LINE_AA) 
            out.write(frame)       
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

