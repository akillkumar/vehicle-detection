'''
    Introduction to Machine Learning
    Monsoon 2020
    Course Project

    Vehicle Detection in Python

    Akhil Kumar
    akhil.kumar_ug21@ashoka.edu.in
'''

import sys
import os
import cv2 as cv

class COLORS:
	clear = '\033[0m'
	blue  = '\033[94m'
	green = '\033[92m'
	cyan  = '\033[96m'
	red   = '\033[91m'
	yell  = '\033[93m'
	mag   = '\033[35m'

# if user provides a video, we want to use that
if len(sys.argv) > 1:
    try:
        # check if its a valid path we can open
        video = cv.VideoCapture (sys.argv[1])
    except:
        # if not, throw an error and quit
        print (COLORS.red + "Unable to load video at " + sys.argv[1] + "\n Usage: python tracker.py <path/to/video>"  + COLORS.clear)
        os._exit (1)
else:
    # default test video
    video = cv.VideoCapture ('vehicles/Urban/march9.avi')

# our pre-trained car classifier
# https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html
classifier = 'classifier.xml'

# Haar cascade classifier
cascade = cv.CascadeClassifier (classifier)

while True:
    # read the curren frame
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-read
    ret, frame = video.read ()

    # if our read was unsuccessful
    if not ret:
        print (COLORS.red + "Video read error" + COLORS.clear)
        os._exit (1)
    
    # convert to grayscale for efficiency
    grayscaled = cv.cvtColor (frame, cv.COLOR_BGR2GRAY)

    # detect cars
    cars = cascade.detectMultiScale (grayscaled)

    #print (cars)

    # the fancy part - draw rectangles    
    for (x, y, w, h) in cars:
        # want to draw it on the regular frame although we detected on grayscale
        # because our final output will be a colored frame
        cv.rectangle (frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    # display the frame with spotted cars
    cv.imshow ('Vehicle Detection', frame)

    # wait for key press
    cv.waitKey (1)




