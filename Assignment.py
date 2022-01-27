import cv2
import numpy as np
import cvzone
import os
from PIL import Image

path = 'Resources/BaggageAI_CV_Hiring_Assignment/threat_images'
path_1 = 'Resources/BaggageAI_CV_Hiring_Assignment/background_images'

images = [] # collection of images
background_images = [] # collection of background images

positions = [[100,100],[50,50],[100,100],[400,400],[250,250]] # positions of where to put objects
colors = [[0, 28, 117, 150, 255, 255]] # mask HSV Values

img_names = os.listdir(path)
bg_names = os.listdir(path_1)

for name in img_names:
    img = cv2.imread(f'{path}/{name}')
    images.append(img)

for name in bg_names:
    img = cv2.imread(f'{path_1}/{name}')
    background_images.append(img)

count = 0
for img, bg_img, posi in zip(images,background_images,positions):
    count += 1
    # Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # Cropping
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]

    # image rotation
    rotated = cvzone.rotateImage(dst,45,0.75)
    imgHSV = cv2.cvtColor(rotated,cv2.COLOR_BGR2HSV)

    # creating mask for threats objects
    lower = np.array(colors[0][0:3])
    upper = np.array(colors[0][3:6])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(rotated, rotated, mask=mask)

    # Creating and applying Transperency mask for threat objects
    alpha = np.sum(imgResult, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    res = np.dstack((imgResult, alpha))
    alpha = np.sum(imgResult, axis=-1) > 0
    alpha = np.uint8(alpha * 200)
    res = np.dstack((imgResult, alpha))

    # Merging Threat object with baggage images
    img = cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA)
    im_pil = Image.fromarray(img)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
    img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2RGBA)
    im_pil_1 = Image.fromarray(img)
    im_pil_1.paste(im_pil,posi,mask=im_pil)
    im_pil_1.show()
    res = cv2.cvtColor(res,cv2.COLOR_BGRA2BGR)
    im_pil_1.save(f'Output_{count}.png')

    cv2.waitKey(0)