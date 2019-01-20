import numpy as np
import cv2
from PIL import Image
import os



# vidcap = cv2.VideoCapture('movie.mov')
# success,image = vidcap.read()
# count = 0
# while success:
#     cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
#     success,image = vidcap.read()
#     print('writing frame ' , count)
#     count += 1





def imageToVector(inputImageLocation):
    img = Image.open(inputImageLocation)
    # img = img.convert('LA')
    width, height = img.size
    imVector = []
    for i in range(0, width):
        arr = []
        for j in range(0, height):
            arr.append(round(img.getpixel((i, j))[0] / 255, 4))
        imVector.append(arr)
    return np.array(imVector).T





fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('video.avi',fourcc,20,(433,413))



def drawBox(maxpos, imArray, refHeight = 98, refWidth = 54):
    for j in range(maxpos[1], maxpos[1]+refWidth):
        imArray[maxpos[0],j] = (255,0,0)
    for j in range(maxpos[1], maxpos[1]+refWidth):
        imArray[maxpos[0]+refHeight,j] = (255,0,0)
    for i in range(maxpos[0], maxpos[0]+refHeight):
        imArray[i,maxpos[1]] = (255,0,0)
    for i in range(maxpos[0], maxpos[0]+refHeight):
        imArray[i,maxpos[1]+refWidth] = (255,0,0)
    return imArray



def exhaustiveSearch(imageName):
    frameVector = imageToVector('frames/' + imageName + '.jpg')
    refVector = imageToVector('reference.jpg')
    frameHeight, frameWidth = len(frameVector), len(frameVector[0])
    refHeight, refWidth = len(refVector), len(refVector[0])

    ravelledReference = refVector.ravel()
    maxScore = np.inf
    maxpos = (0,0)
    for i in range(0, frameHeight-refHeight):
        for j in range(0, frameWidth-refWidth):
            p,q = 0,0
            block = frameVector[i:i+refHeight, j:j+refWidth].ravel()
            score = np.correlate(block, ravelledReference)
            if(score[0] < maxScore):
                maxScore = score[0]
                maxpos = (i,j)

    img = Image.open('frames/' + imageName + '.jpg')
    imArray = np.array(img)
    imArray = drawBox(maxpos, imArray, refHeight, refWidth)
    img = Image.fromarray(imArray, 'RGB')


    img.save('merge_frames/'+imageName +'.jpg')
    img = cv2.imread('merge_frames/'+imageName +'.jpg')
    video.write(img)
    return maxpos









def logarithmicSearch(param):



    prev = exhaustiveSearch('frame0')

    refVector = imageToVector('reference.jpg')
    refHeight, refWidth = len(refVector), len(refVector[0])



    for fr in range(1, 20):
        imageName = 'frame' + str(fr)
        pPower = 2
        p = param
        while True:
            k = np.ceil(np.log2(p))
            d = 2 ** ((int(k)-1))
            if d < 1 : break
            x, y = prev
            direction = [[x,y],[x+d,y],[x-d,y],[x,y+d],[x,y-d],[x+d,y+d],[x+d,y-d],[x-d,y+d],[x-d,y-d]]
            frameVector = imageToVector('frames/' + imageName + '.jpg')

            ravelledReference = refVector.ravel()
            maxScore = np.inf
            maxpos = (x, y)
            for dir in direction:
                block = frameVector[dir[0]:dir[0] + refHeight, dir[1]:dir[1] + refWidth].ravel()
                score = np.correlate(block, ravelledReference)
                if (score[0] < maxScore):
                    maxScore = score[0]
                    maxpos = dir
            p/= (2.0 ** pPower)
            pPower += 1
            prev = maxpos

        img = Image.open('frames/' + imageName + '.jpg')
        imArray = np.array(img)



        imArray = drawBox(prev, imArray, refHeight, refWidth)



        img = Image.fromarray(imArray, 'RGB')

        img.save('merge_frames/' + imageName + '.jpg')
        img = cv2.imread('merge_frames/' + imageName + '.jpg')
        video.write(img)
        print('# logsearch done frame = ', fr)






def hierarchical():
    pass




'''
exhaustive test
'''
# totalFrames = count
# for i in range(0, 20):
#     exhaustiveSearch('frame'+str(i))
#     print('# image ', i  , ' done!')



logarithmicSearch(7)







cv2.destroyAllWindows()
video.release()























