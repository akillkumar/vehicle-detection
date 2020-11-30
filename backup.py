import cv2 as cv

# get our pre-trained classifier
classifier = 'classifier.xml'
tracker = cv.CascadeClassifier (classifier)

# pick a random test image
img_file = 'test_image.jpg'
img = cv.imread (img_file)
bw  = cv.cvtColor (img, cv.COLOR_BGR2GRAY) # convert to black and white for better speed

# use our classifier to detect cars
cars = tracker.detectMultiScale (bw)

# and draw some boxes
for (x, y, w, h) in cars:
    cv.rectangle (img, (x, y), (x+w, y+h),)


cv.imshow ('Car detector', img)
cv.waitKey()

print ("Done")