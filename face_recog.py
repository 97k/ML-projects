import argparse
from scipy.misc import imsave
import cognitive_face as CF
import cv2
from sys import platform


# -------- Using microsoft cognitive face api --------------------

def recognize(key):

  CF.Key.set(key) # set API key
  BASE_URL = 'https://westus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
  CF.BaseUrl.set(BASE_URL)


  while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if platform.startswith('win'): # for windows we don't display video due to camera issues
      cap.release()
    imsave('tmp.png', frame)

    result = CF.face.detect('tmp.png', attributes='age,gender')
    try:
      for face in result:
        gender = face['faceAttributes']['gender']
        age = face['faceAttributes']['age']
        print (gender, age)
        if platform != 'darwin': # for mac we display the video, face bounding box, age & gender
          rect = face['faceRectangle']
          width = rect['width']
          top = rect['top']
          height = rect['height']
          left = rect['left']
          cv2.rectangle(frame, (left, top), (left + width, top + height),
                        (0, 255, 0), 2)
          cv2.putText(frame, '{},{}'.format(gender, int(age)), (left, top),
                      cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
          cv2.imshow('Demo', frame)

    except not result:
      continue

    except KeyboardInterrupt:
      cap.release()
      cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
#   parser.add_argument('-k','--key', required=True, type=str,
#                       help='key for face api')
  args = parser.parse_args()

#   recognize(args.key)
  recognize('your-key-here')



# ------------------------------------- Using OpenCV -------------------------------------


import numpy as np

import cv2


# opencv-3.4.0/data/haarcascades/

face_cascade = cv2.CascadeClassifier('/home/aadi/opencv-3.4.0/sources/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/home/aadi/opencv-3.4.0/sources/data/haarcascades/haarcascade_eye.xml')

# cap = cv2.VideoCapture(0)<br>while 1:
cap = cv2.VideoCapture("/home/aadi/Desktop/OpenCV Python TUTORIAL.mp4")
while True:


    ret, img = cap.read()

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(img, 0)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]

        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    print ("found " +str(len(faces)) +" face(s)")

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:

        break

cap.release()

cv2.destroyAllWindows()

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascades_frontalface_alt2')
cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	# face cascade works in gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbours=5)

	for  (x, y, z, h) in faces:
		print(x, y, z, h)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.distroyAllWindows()	