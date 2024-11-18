#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: okokprojects
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        
        # load model
        model = load_model('model_plant.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (300, 300))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        classes = np.argmax(result)
        print("", classes)

        if   classes <=0:
            prediction = 'Identified Plant is Aloevera,Aloe vera can be used topically to treat burns, wounds, acne, psoriasis, mouth sores, ulcers, lichen planus, and radiation-induced skin toxicity'
            print("Classification result", prediction)

        elif classes <=1:
            prediction = 'Identified Plant is Amla,Skin conditions: Amla has anti-inflammatory, antioxidant, and antibacterial properties, and can help with acne, eczema, and other skin conditions'
            print("Classification result", prediction)
        elif classes <=2:
            prediction = 'Identified Plant is Betel,Gastrointestinal disorders: Betel leaves can help treat gastrointestinal disorders, ease flatulence, and improve digestion.'
            print("Classification result", prediction)
        elif classes <= 3:
            prediction = 'Identified Plant is Castor,Used for rheumatic conditions, gastropathy, constipation, inflammation, fever, ascites, bronchitis, cough, skin diseases, colic, and lumbago'
            print("Classification result", prediction)
        elif classes <=4:
            prediction = 'Identified Plant is Mint,Digestion: Mint is a popular remedy for digestive problems. Peppermint oil in capsules may help with belly pain from irritable bowel syndrome.'
            print("Classification result", prediction)
        elif classes <=5:
            prediction = 'Identified Plant is Neem,Neem preparations are reportedly efficacious against a variety of skin diseases, septic sores, and infected burns. The leaves, applied in the form of poultices or decoctions, are also recommended for boils, ulcers, and eczema'
            print("Classification result", prediction)
        elif classes <=6:
            prediction = 'Identified Plant is Tulasi,Antimicrobial: Tulsi has antimicrobial properties, including antibacterial, antiviral, antifungal, antiprotozoal, antimalarial, and anthelmintic'
            print("Classification result", prediction)

        else:
            prediction = 'None'
            print("Classification result", prediction)


        return [prediction]


