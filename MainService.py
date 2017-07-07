# -*- coding: utf-8 -*-
"""
Created on Fri 07 Jul 2017

@author: Amr
@email : amr.jawwad@outlook.com
"""

import numpy as np
import sys
from sklearn.externals import joblib
import urllib, PIL, cStringIO
import json

def DigitRecogService(input_json):
    #Loading input JSON file
    try:
        Input = json.load(open(input_json))
    except:
        print "Could not load input JSON file. Terminating service."
        sys.exit(0)
    
    Output =[]
    
    #Loading pre-trained Classifier
    try:
        Classifier = joblib.load('SVM_model.pkl')
    except:
        print "Could not load classifier file. Terminating service."
        sys.exit(0)
    
    #Main loop, iterates on the URLs in JSON input
    for s in Input["ImgURLs"]:
        #Opening image
        print "Processing image:"
        print s
        try:
            Img = PIL.Image.open(cStringIO.StringIO(urllib.urlopen(s).read()))
        except:
            print "There was a problem with loading image with URL:"
            print s
            print "Moving on to next image."
            continue
        
        #Image pre-processing:
        #1. Resizing to MNIST size 28x28
        #ASSUMPTION: the image's aspect ratio is not so far from a square
        Img =Img.resize((28,28),PIL.Image.ANTIALIAS)
        #2. Converting to grayscale
        Img = Img.convert('LA')
        ImgAsArray = np.array(Img.getdata(0))
        #3. Inverting if the background is white, and the number is black
        #ASSUMPTION: the image is mostly background
        hist = Img.histogram()
        if hist.index(max(hist)) > 127:
            ImgAsArray = 255 - ImgAsArray
            print "Image inverted"
        ImgAsArray = ImgAsArray/255.0
        
        #Predict the digit
        Output.append(Classifier.predict(ImgAsArray.reshape(1,-1))[0])
    #End of main loop
    
    #Write output to JSON file
    print "Processing completed."
    try:
        json.dump({"Output":Output},open('Output.json', 'w'))
        return json.dumps({"Output":Output})
    except:
        print "Could not write output to JSON file. Displaying output:"
        print Output
        
DigitRecogService('Input.json')