import cv2
import numpy as np
import matplotlib.pyplot as plt



def Drawing():
    def draw_circle(event, x, y, drawing = False, mode=False, ix=-1, iy=-1):

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:            
                cv2.circle(img, (x, y), 10, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img, (x, y), 10, (255, 255, 255), -1)


    img = np.zeros((256, 256, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle) 

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    #Transfer image to gray-scale level and fit the size of MNIST dataset
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.GaussianBlur(imgGray, (15,15), 0)
    plt.imshow(imgGray)
    plt.show()
    imgGray = cv2.resize(imgGray, (28,28))
    #cv2.imshow('MNIST-size Gray Image', imgGray)
    #cv2.waitKey()
    return imgGray

