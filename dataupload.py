import os
import tkinter as tk
import time
from random import shuffle

import PIL.Image as pimg
import PIL.ImageTk as pimgtk

from tkinter import *
from tkinter import messagebox, filedialog
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from keras.preprocessing.image import ImageDataGenerator
import tk as tk
import tkinter as tk

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
batch_size = 32


from PIL import Image, ImageTk
from glob import glob

from PIL.ImageFile import ImageFile
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model


class ViewData:
    def __init__(self):
        def dataupload():
            s1 = "E:\Medicinal_plants\Dataset"
            messagebox.showinfo("Success", s1)

        def viewdata():
            workingDir = "E:\Medicinal_plants\Dataset"
            PATH = os.path.sep.join([workingDir, "Train"])
            train_dir = os.path.join("E:\Medicinal_plants\Dataset", "Train")

            # Getting the path to the validation directory
            validation_dir = os.path.join("E:\Medicinal_plants\Dataset", "Train")

            train_dir1 = os.path.join(train_dir, "Aloevera")
            train_dir2 = os.path.join(train_dir, "Amla")
            train_dir3 = os.path.join(train_dir, "Betel")
            train_dir4 = os.path.join(train_dir, "Castor")
            train_dir5 = os.path.join(train_dir, "Mint")
            train_dir6 = os.path.join(train_dir, "Neem")
            train_dir7 = os.path.join(train_dir, "Tulasi")
            train_dir8 = os.path.join(train_dir, "wNewImg")

            # Getting the path to the directory for the parasitized validation cell images and
            # the path to the directory for the uninfected validation cell images
            val_dir1 = os.path.join(validation_dir, "Aloevera")
            val_dir2 = os.path.join(validation_dir, "Amla")
            val_dir3 = os.path.join(validation_dir, "Betel")
            val_dir4 = os.path.join(validation_dir, "Castor")
            val_dir5 = os.path.join(validation_dir, "Mint")
            val_dir6 = os.path.join(validation_dir, "Neem")
            val_dir7 = os.path.join(validation_dir, "Tulasi")
            val_dir8 = os.path.join(validation_dir, "wNewImg")
            # Getting the number of images present in the parasitized training directory and the
            # number of images present in the uninfected training directory
            train_images1 = len(os.listdir(train_dir1))
            train_images2 = len(os.listdir(train_dir2))
            train_images3 = len(os.listdir(train_dir3))
            train_images4 = len(os.listdir(train_dir4))
            train_images5 = len(os.listdir(train_dir5))
            train_images6 = len(os.listdir(train_dir6))
            train_images7 = len(os.listdir(train_dir7))
            train_images8 = len(os.listdir(train_dir8))

            # Getting the number of images present in the parasitized validation directory and the
            # number of images present in the uninfected validation directory
            images_val1 = len(os.listdir(val_dir1))
            images_val2 = len(os.listdir(val_dir2))
            images_val3 = len(os.listdir(val_dir3))
            images_val4 = len(os.listdir(val_dir4))
            images_val5 = len(os.listdir(val_dir5))
            images_val6 = len(os.listdir(val_dir6))
            images_val7 = len(os.listdir(val_dir7))
            images_val8 = len(os.listdir(val_dir8))

            # Getting the sum of both the training images and validation images
            total_train =  train_images1 + train_images2+train_images3+train_images4+train_images5+train_images6+train_images7+train_images8
            total_val = images_val1 + images_val2+images_val3+images_val4+images_val5+images_val6+images_val7+images_val8

            print("Total Train Images: {}".format(total_train));
            #print("Total Validation: {}".format(total_val));
            s1 = "Total Train Image "+format(total_train)
            messagebox.showinfo("Success", s1)
           # messagebox.showinfo("Extraction Sucessfully")

        def viewdata1():
            workingDir = "E:\Medicinal_plants\Dataset"
            PATH = os.path.sep.join([workingDir, "Train"])

            validation_dir = os.path.join("E:\Medicinal_plants\Dataset", "Train")
            train_dir = os.path.join("E:\Medicinal_plants\Dataset", "Train")
            batch_size = 2000
            epochs = 20
            IMG_HEIGHT = 98
            IMG_WIDTH = 98

            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                shuffle=True
                                                                )

            validation_datagen = ImageDataGenerator(rescale=1. / 255)

            validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                          batch_size=batch_size,
                                                                          class_mode='binary',
                                                                          shuffle=True
                                                                          )
            s1 = "Total Class Extracted Sucessfully"
            messagebox.showinfo(" Success", s1)
        def build():
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            train_path = 'Dataset/Train'
            valid_path = 'Dataset/Test'

            # load model without output layer

            IMAGE_SIZE = [300, 300]
            # add preprocessing layer to the front of VGG
            vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

            # don't train existing weights
            for layer in vgg.layers:
                layer.trainable = False

            # useful for getting number of classes
            folders = glob('Dataset/Train/*')

            x = Flatten()(vgg.output)
            # x = Dense(1000, activation='relu')(x)
            prediction = Dense(len(folders), activation='softmax')(x)

            # create a model object
            model = Model(inputs=vgg.input, outputs=prediction)

            # view the structure of the model
            model.summary()

            # tell the model what cost and optimization method to use
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # Use the Image Data Generator to import the images from the

            from keras.preprocessing.image import ImageDataGenerator
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            val_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                             target_size=(300, 300),
                                                             batch_size=32,
                                                             class_mode='categorical')

            val_set = val_datagen.flow_from_directory('Dataset/Test',
                                                      target_size=(300, 300),
                                                      batch_size=32,
                                                      class_mode='categorical')

            # fit the model
            history = model.fit_generator(training_set,
                                          validation_data=val_set,
                                          epochs=5,
                                          steps_per_epoch=len(training_set),
                                          validation_steps=len(val_set))

            model.save('model_plant.h5')
            s1="Model Build Sucessfully"
            messagebox.showinfo(" Success",s1)


           # print(pd.DataFrame(history.history))

           # pd.DataFrame(history.history).plot()





        win = Tk()
        win.title("Identification of Different Medical Plants")
        win.maxsize(width=900, height=800)
        win.minsize(width=900, height=800)
        win.configure(bg='#99ddff')

        image1 = Image.open("1.jpg")
        img = image1.resize((900, 800))

        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(win, image=test)
        label1.image = test


        # Position image
        label1.place(x=1, y=1)

        # image1 = Image.open("3.png")
        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(win, image=test)
        label1.image = test

        Label(win, text='Identification of Different Medical Plants', bg="#34bfbb", font='verdana 15 bold') \
            .place(x=170, y=120)
        btnbrowse = Button(win, text="Dataset source", font=' Verdana 10 bold', command=lambda: dataupload())
        btnbrowse.place(x=70, y=200)

        btncamera = Button(win, text="Trained Image Extraction", font='Verdana 10 bold', command=lambda: viewdata())
        btncamera.place(x=220, y=200)

        btnsend = Button(win, text="Total Class Extraction", font='Verdana 10 bold', command=lambda: viewdata1())
        btnsend.place(x=450, y=200)

        btnsend = Button(win, text="Build CNN Model", font='Verdana 10 bold', command=lambda: build())
        btnsend.place(x=650, y=200)

        win.mainloop()
