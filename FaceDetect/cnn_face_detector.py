
# coding: utf-8

# In[3]:

import sys
import dlib
import cv2

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])

for f in sys.argv[2:]:
    
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    
    # Detection
    dets = cnn_face_detector(img, 1)
    
    print("Number of faces detected: {}".format(len(dets)))

    for i, d in enumerate(dets):
        face = d.rect
        print("Detection {}: Left: {} Top: {} Bottom: {} Confidence: {}".format(i, face.left(), face.top(), face.right(), d.rect.bottom(), d.confidence))
        
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        
        cv2.rectangle(img, (left,top), (right,bottom), (0,255,0), 3)
        cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f, img)
        
k = cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:



