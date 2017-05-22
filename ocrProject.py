import numpy as np
import cv2

print("Enter Input File Name")
file_out = input()

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

file0="digits_gray.png"
cv2.imwrite(file0,gray)


cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=1)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy

# save the data
np.savez('ocr_knn_data.npz',train=train, train_labels=train_labels)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 32:
            break  # esc to quit

show_webcam(mirror=False)


# Camera 0 is the integrated web cam on my netbook
camera_port = 0

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(0)

# Captures a single image from the camera and returns it in PIL format
def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im

# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
camera_capture = get_image()
#cv2.destroyAllWindows()
file = "test_image.png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
cv2.imwrite(file, camera_capture)

# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
del (camera)

gray_image=cv2.cvtColor(camera_capture, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

_,threshold_image = cv2.threshold(blurred_image,70,255,cv2.THRESH_BINARY_INV)
file2 = "threshold_image.png"
cv2.imwrite(file2,threshold_image)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
#threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

cnts,_ = cv2.findContours(threshold_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
iter = 1

for c in cnts:
    if cv2.contourArea(c) > 5000:
        [x, y, w, h]= cv2.boundingRect(c)

        roi = threshold_image[y-int(0.2*h):y + int(1.2*h), x-int(0.2*w):x + int(1.2*w)]

        resized_image = cv2.resize(roi, (20, 20), 0, 0, interpolation=cv2.INTER_AREA)
        file3 = ("resize_image%d.png" % iter)
        cv2.imwrite(file3, resized_image)
        _, rethreshold_image = cv2.threshold(resized_image, 50, 255, cv2.THRESH_BINARY)
        file4 = ("rethreshold_image%d.png" % iter)
        cv2.imwrite(file4, rethreshold_image)
        resized_image_string = rethreshold_image.reshape(-1, 400).astype(np.float32)
        file5 = ("image_string%d.png" % iter)
        cv2.imwrite(file5, resized_image_string)
        finalRet, finalResult, finalNeighbors, finalDist = knn.find_nearest(resized_image_string, k=10)

        print ("contour# %d" % iter)
        print ("contour Area %d" % cv2.contourArea(c))
        print ("Return: %s" % finalRet)
        print ("Result: %s" % finalResult)
        print ("Neighbors: %s" % finalNeighbors)
        print ("Distance: %s" % finalDist)

        with open(file_out, "r+") as fout:
            fout.write(str(finalRet))

        iter+=1

fout.close()