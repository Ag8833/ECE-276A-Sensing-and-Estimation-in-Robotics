from roipoly import roipoly 
from PIL import Image
import pylab as pl
import numpy as np
import json
import cv2
import os

# Folders for trainset images and testset images.
trainingFolder = "trainset"
testingFolder = "testset"

# Preloading the training output to speed up computation time.
redPixelData = np.loadtxt('FinalizedTrainingOutputRed.txt',delimiter=',')
brownPixelData = np.loadtxt('FinalizedTrainingOutputBrown.txt',delimiter=',')
grayPixelData = np.loadtxt('FinalizedTrainingOutputGray.txt',delimiter=',')

# Arrays for the pixel mean values, converting to numpy arrays for faster computation.
redPixelMean = [0,0,0]
redPixelMean = np.array(redPixelMean)
brownPixelMean = [0,0,0]
brownPixelMean = np.array(brownPixelMean)
grayPixelMean = [0,0,0]
grayPixelMean = np.array(grayPixelMean)
currentSumR = 0
currentSumG = 0
currentSumB = 0

# Arrays for the pixel variance values, converting to numpy arrays for faster computation.
redPixelVariance = [0,0,0]
redPixelVariance = np.array(redPixelVariance)
brownPixelVariance = [0,0,0]
brownPixelVariance = np.array(brownPixelVariance)
grayPixelVariance = [0,0,0]
grayPixelVariance = np.array(grayPixelVariance)

# Preloading this to speed up computation time.
redPixelCovariance = np.loadtxt('RedPixelCovariance.txt',delimiter=',')
brownPixelCovariance = np.loadtxt('BrownPixelCovariance.txt',delimiter=',')
grayPixelCovariance = np.loadtxt('GrayPixelCovariance.txt',delimiter=',')

# Precalculation of inverse and determinants to speed up computation time.
redPixelCovarianceInverse = np.linalg.inv(redPixelCovariance)
redPixelCovarianceDeterminant = np.linalg.det(redPixelCovariance)
brownPixelCovarianceInverse = np.linalg.inv(brownPixelCovariance)
brownPixelCovarianceDeterminant = np.linalg.det(brownPixelCovariance)
grayPixelCovarianceInverse = np.linalg.inv(grayPixelCovariance)
grayPixelCovarianceDeterminant = np.linalg.det(grayPixelCovariance)

# Preloading of barrel distance data.
barrelDistance = np.loadtxt('BarrelDistanceData.txt',delimiter=',')
m = 0
b = 0

# Matrix to be used by BarrelDetection function
maskMatrix = np.empty([1200,900])

# Function that will generate the pixel data for the trainset images by allowing the user to select a 
# region of color for Red, Brown and Gray. It will then output the pixel data to TrainsetOutput.txt.
def PixelDataGenerator():
    redPixelValues = []
    brownPixelValues = []
    grayPixelValues = []

    for filename in os.listdir(trainingFolder):
        img = Image.open(os.path.join(trainingFolder,filename))
        pix = img.load()
        
        ######### RED ROI #########
        pl.imshow(img, interpolation='nearest', cmap="Greys")
        pl.title("left click: line segment         right click: close region")
        ROI1 = roipoly(roicolor='red')
        
        xPointsAverage = int((ROI1.allxpoints[0] + ROI1.allxpoints[1] + ROI1.allxpoints[2] + ROI1.allxpoints[3])/4)
        yPointsAverage = int((ROI1.allypoints[0] + ROI1.allypoints[1] + ROI1.allypoints[2] + ROI1.allypoints[3])/4)
        print("Red Coordinates - ")
        print(xPointsAverage)
        print(yPointsAverage)
        print("Red Value - ")
        redPixelValues.append(pix[xPointsAverage,yPointsAverage])
        print(redPixelValues)
        print("")
        ######### RED ROI #########
        
        ######### BROWN ROI #########
        brownResponse = input('Press 1 if there is brown, 0 if not: ')
        if brownResponse is 1:
            pl.imshow(img, interpolation='nearest', cmap="Greys")
            ROI1.displayROI()
            pl.title('draw second ROI')
            ROI2 = roipoly(roicolor='brown')
            
            xPointsAverage = int((ROI2.allxpoints[0] + ROI2.allxpoints[1] + ROI2.allxpoints[2] + ROI2.allxpoints[3])/4)
            yPointsAverage = int((ROI2.allypoints[0] + ROI2.allypoints[1] + ROI2.allypoints[2] + ROI2.allypoints[3])/4)
            print("Brown Coordinates - ")
            print(xPointsAverage)
            print(yPointsAverage)
            print("Brown Value - ")
            brownPixelValues.append(pix[xPointsAverage,yPointsAverage])
            print(brownPixelValues)
            print("")
        ######### BROWN ROI #########
        
        ######### GRAY ROI #########
        grayResponse = input('Press 1 if there is gray, 0 if not: ')
        if grayResponse is 1:
            pl.imshow(img, interpolation='nearest', cmap="Greys")
            ROI1.displayROI()
            if brownResponse is 1: ROI2.displayROI()
            pl.title('draw third ROI')
            ROI3 = roipoly(roicolor='gray')
            
            xPointsAverage = int((ROI3.allxpoints[0] + ROI3.allxpoints[1] + ROI3.allxpoints[2] + ROI3.allxpoints[3])/4)
            yPointsAverage = int((ROI3.allypoints[0] + ROI3.allypoints[1] + ROI3.allypoints[2] + ROI3.allypoints[3])/4)
            print("Gray Coordinates - ")
            print(xPointsAverage)
            print(yPointsAverage)
            print("Gray Value - ")
            grayPixelValues.append(pix[xPointsAverage,yPointsAverage])
            print(grayPixelValues)
            print("")
        ######### GRAY ROI #########
        
        f = open('TrainsetOutput.txt', 'w')
        json.dump("RED - ", f)
        json.dump(redPixelValues, f)
        json.dump("BROWN - ", f)
        json.dump(brownPixelValues, f)
        json.dump("GRAY - ", f)
        json.dump(grayPixelValues, f)
        f.close()
        
# Function to generate the mean of the pixel data based on the redPixelData, brownPixelData 
# and grayPixelData arrays from the PixelDataGenerator function.
def MeanPixelDataCalculator():
    ######### RED ROI #########
    currentSumR = currentSumG = currentSumB = 0
    for i in range(0,len(redPixelData)):
        currentSumR = currentSumR + redPixelData[i,0] 
        currentSumG = currentSumG + redPixelData[i,1] 
        currentSumB = currentSumB + redPixelData[i,2] 
    redPixelMean[0] = round((currentSumR / len(redPixelData)),2)
    redPixelMean[1] = round((currentSumG / len(redPixelData)),2)
    redPixelMean[2] = round((currentSumB / len(redPixelData)),2)
    ######### RED ROI #########

    ######### BROWN ROI #########
    currentSumR = currentSumG = currentSumB = 0
    for i in range(0,len(brownPixelData)):
        currentSumR = currentSumR + brownPixelData[i,0] 
        currentSumG = currentSumG + brownPixelData[i,1] 
        currentSumB = currentSumB + brownPixelData[i,2] 
    brownPixelMean[0] = round((currentSumR / len(brownPixelData)),2)
    brownPixelMean[1] = round((currentSumG / len(brownPixelData)),2)
    brownPixelMean[2] = round((currentSumB / len(brownPixelData)),2)
    ######### BROWN ROI #########

    ######### GRAY ROI #########
    currentSumR = currentSumG = currentSumB = 0
    for i in range(0,len(grayPixelData)):
        currentSumR = currentSumR + grayPixelData[i,0] 
        currentSumG = currentSumG + grayPixelData[i,1] 
        currentSumB = currentSumB + grayPixelData[i,2] 
    grayPixelMean[0] = round((currentSumR / len(grayPixelData)),2)
    grayPixelMean[1] = round((currentSumG / len(grayPixelData)),2)
    grayPixelMean[2] = round((currentSumB / len(grayPixelData)),2)
    ######### GRAY ROI #########

# Function to print out the mean pixel data
def PrintMeanData():
    print"MEAN DATA RBG - "
    print"RED - "
    print(redPixelMean)
    print"BROWN - "
    print(brownPixelMean)
    print"GRAY - "
    print(grayPixelMean)
    print" "
    
# Function to generate the variance of the pixel data based on the redPixelData, brownPixelData 
# and grayPixelData arrays from the PixelDataGenerator function.
def PopulationVarianceCalculator():
    ######### RED ROI #########
    currentSumR = currentSumG = currentSumB = 0;
    for i in range(0,len(redPixelData)):
        currentSumR = currentSumR + ((redPixelData[i,0] - redPixelMean[0]) ** 2)
        currentSumG = currentSumG + ((redPixelData[i,1] - redPixelMean[1]) ** 2)
        currentSumB = currentSumB + ((redPixelData[i,2] - redPixelMean[2]) ** 2)
    redPixelVariance[0] = round((currentSumR / len(redPixelData)),2)
    redPixelVariance[1] = round((currentSumG / len(redPixelData)),2)
    redPixelVariance[2] = round((currentSumB / len(redPixelData)),2)
    ######### RED ROI #########
    
    ######### BROWN ROI #########
    currentSumR = currentSumG = currentSumB = 0
    for i in range(0,len(brownPixelData)):
        currentSumR = currentSumR + ((brownPixelData[i,0] - brownPixelMean[0]) ** 2) 
        currentSumG = currentSumG + ((brownPixelData[i,1] - brownPixelMean[1]) ** 2)
        currentSumB = currentSumB + ((brownPixelData[i,2] - brownPixelMean[2]) ** 2)
    brownPixelVariance[0] = round((currentSumR / len(brownPixelData)),2)
    brownPixelVariance[1] = round((currentSumG / len(brownPixelData)),2)
    brownPixelVariance[2] = round((currentSumB / len(brownPixelData)),2)
    ######### BROWN ROI #########

    ######### GRAY ROI #########
    currentSumR = currentSumG = currentSumB = 0
    for i in range(0,len(grayPixelData)):
        currentSumR = currentSumR + ((grayPixelData[i,0] - grayPixelMean[0]) ** 2)
        currentSumG = currentSumG + ((grayPixelData[i,1] - grayPixelMean[1]) ** 2)
        currentSumB = currentSumB + ((grayPixelData[i,2] - grayPixelMean[2]) ** 2)
    grayPixelVariance[0] = round((currentSumR / len(grayPixelData)),2)
    grayPixelVariance[1] = round((currentSumG / len(grayPixelData)),2)
    grayPixelVariance[2] = round((currentSumB / len(grayPixelData)),2)
    ######### GRAY ROI #########
    
# Function to print out the variance pixel data    
def PrintPopulationVarianceData():
    print"VARIANCE DATA RBG - "
    print"RED - "
    print(redPixelVariance)
    print"BROWN - "
    print(brownPixelVariance)
    print"GRAY - "
    print(grayPixelVariance)
    print" "
    
# Function to generate the covariance of the pixel data based on the redPixelData, brownPixelData 
# and grayPixelData arrays from the PixelDataGenerator function.
def CovarianceCalculator():
    tempMatrix = [[0,0,0],[0,0,0],[0,0,0]]
    tempMatrix = np.array(tempMatrix)
    
    tempMatrixTranspose = [[0,0,0],[0,0,0],[0,0,0]]
    tempMatrixTranspose = np.array(tempMatrixTranspose)
    
    tempRedPixelCovariance = [[0,0,0],[0,0,0],[0,0,0]]
    tempRedPixelCovariance = np.array(tempRedPixelCovariance)
    tempBrownPixelCovariance = [[0,0,0],[0,0,0],[0,0,0]]
    tempBrownPixelCovariance = np.array(tempBrownPixelCovariance)
    tempGrayPixelCovariance = [[0,0,0],[0,0,0],[0,0,0]]
    tempGrayPixelCovariance = np.array(tempGrayPixelCovariance)
    
    ######### RED ROI #########
    for i in range(0, len(redPixelData)):
        for j in range(0, len(redPixelMean)):
            for k in range(0, len(redPixelMean)):
                tempMatrix[j,k] = redPixelData[i,k] - redPixelMean[j]
                         
        tempMatrixTranspose = np.transpose(tempMatrix)
        if i == 0:
            tempRedPixelCovariance = np.dot(tempMatrix , tempMatrixTranspose)
        else:
            tempRedPixelCovariance = np.add(tempRedPixelCovariance, np.dot(tempMatrix, tempMatrixTranspose))
         
    for i in range(0, len(tempRedPixelCovariance)):
        for j in range(0, len(tempRedPixelCovariance)):
            tempRedPixelCovariance[i,j] = tempRedPixelCovariance[i,j] / len(redPixelData)
            redPixelCovariance[i,j] = abs(np.copy(tempRedPixelCovariance[i,j]))
    ######### RED ROI #########
     
    ######### BROWN ROI #########
    for i in range(0, len(brownPixelData)):
        for j in range(0, len(brownPixelMean)):
            for k in range(0, len(brownPixelMean)):
                tempMatrix[j,k] = brownPixelData[i,k] - brownPixelMean[j]
                         
        tempMatrixTranspose = np.transpose(tempMatrix)
        if i == 0:
            tempBrownPixelCovariance = np.dot(tempMatrix , tempMatrixTranspose)
        else:
            tempBrownPixelCovariance = np.add(tempBrownPixelCovariance, np.dot(tempMatrix, tempMatrixTranspose))
         
    for i in range(0, len(tempBrownPixelCovariance)):
        for j in range(0, len(tempBrownPixelCovariance)):
            tempBrownPixelCovariance[i,j] = tempBrownPixelCovariance[i,j] / len(brownPixelData)
            brownPixelCovariance[i,j] = abs(np.copy(tempBrownPixelCovariance[i,j]))
    ######### BROWN ROI #########
     
    ######### GRAY ROI #########
    for i in range(0, len(grayPixelData)):
        for j in range(0, len(grayPixelMean)):
            for k in range(0, len(grayPixelMean)):
                tempMatrix[j,k] = grayPixelData[i,k] - grayPixelMean[j]
                         
        tempMatrixTranspose = np.transpose(tempMatrix)
        if i == 0:
            tempGrayPixelCovariance = np.dot(tempMatrix , tempMatrixTranspose)
        else:
            tempGrayPixelCovariance = np.add(tempGrayPixelCovariance, np.dot(tempMatrix, tempMatrixTranspose))
         
    for i in range(0, len(tempGrayPixelCovariance)):
        for j in range(0, len(tempGrayPixelCovariance)):
            tempGrayPixelCovariance[i,j] = tempGrayPixelCovariance[i,j] / len(grayPixelData)
            grayPixelCovariance[i,j] = abs(np.copy(tempGrayPixelCovariance[i,j]))
    ######### GRAY ROI #########
    
# Function to print out the covariance pixel data    
def PrintCovarianceData():
    print"RED - "
    print(redPixelCovariance)
    print""
    print"BROWN - "
    print(brownPixelCovariance)
    print""
    print"GRAY - "
    print(grayPixelCovariance)
    print" "
    
# Function to calculate the Single Gaussian BDR. It will compute values for probability of 
# red, brown and gray. Then it will return 1 if red is the smallest, otherwise 0.
def SingleGaussianTest(samplePixel): 
    ######### RED ROI #########
    x = 1/np.dot(((2*3.14) ** (3/2)),(redPixelCovarianceDeterminant))
    xa = (-1/2)*np.transpose(np.subtract(samplePixel,redPixelMean))
    xb = redPixelCovarianceInverse
    xc = np.subtract(samplePixel,redPixelMean)
    xd = np.dot(np.dot(xa,xb),xc)
    probabilityRed = x * xd
    ######### RED ROI #########
    
    ######### BROWN ROI #########
    x = 1/np.dot(((2*3.14) ** (3/2)),(brownPixelCovarianceDeterminant))
    xa = (-1/2)*np.transpose(np.subtract(samplePixel,brownPixelMean))
    xb = brownPixelCovarianceInverse
    xc = np.subtract(samplePixel,brownPixelMean)
    xd = np.dot(np.dot(xa,xb),xc)
    probabilityBrown = x * xd
    ######### BROWN ROI #########
    
    ######### GRAY ROI #########
    x = 1/np.dot(((2*3.14) ** (3/2)),(grayPixelCovarianceDeterminant))
    xa = (-1/2)*np.transpose(np.subtract(samplePixel,grayPixelMean))
    xb = grayPixelCovarianceInverse
    xc = np.subtract(samplePixel,grayPixelMean)
    xd = np.dot(np.dot(xa,xb),xc)
    probabilityGray = x * xd
    ######### GRAY ROI #########
    
    if (abs(probabilityRed) < abs(probabilityBrown) and (abs(probabilityRed) < abs(probabilityGray))):
        return 1
    elif (abs(probabilityBrown) < abs(probabilityRed) and (abs(probabilityBrown) < abs(probabilityGray))):
        return 0
    elif (abs(probabilityGray) < abs(probabilityRed) and (abs(probabilityGray) < abs(probabilityBrown))):
        return 0
    else:
        return 0
    
# Function to calculate the equation for the Linear Regression of barrel distance.
def BarrelLinearRegression():
    global m
    global b
    area = [row[0] for row in barrelDistance]
    distance = [row[1] for row in barrelDistance]
    covariance = 0
    
    meanArea = sum(area) / float(len(area))
    meanDistance = sum(distance) / float(len(distance))
    varianceArea = sum([(i - meanArea)**2 for i in area])
    varianceDistance = sum([(i - meanDistance)**2 for i in distance])
    for i in range(len(area)):
        covariance += (area[i] - meanArea) * (distance[i] - meanDistance)
    
    m = covariance/varianceArea
    b = meanDistance - m * meanArea

# Function to iterate through images in testingFolder, for each image it will take every pixel and using
# the SingleGaussianTest function it will determine whether that pixel is red or not. After it checks every
# pixel it will create a maskMatrix and display it, showing which pixels are red and which are not. From the
# maskMatrix it will find the contours of the matrix, taking the contour with the largest arcLength as the 
# barrel region. Once it has this contour it will create a bounding rectangle for the contour, generating the
# coordinates of the four corners in order to draw the rectangle around the barrel. Lastly it will compute
# the area of this barrel, and then using the coefficients determined by BarrelLinearRegression it will 
# estimate the distance the barrel is at.
def BarrelDetection():
    testingOutputFolder = open('TestsetOutput.txt', 'w')
    maskMatrix = np.empty([900,1200])
    for filename in os.listdir(testingFolder):
        img = cv2.imread(os.path.join(testingFolder,filename))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for i in range(0, (900)):
            for j in range(0, (1200)):
                print(i)
                pixelValue = rgb[i,j]
                maskMatrix[i,j] = SingleGaussianTest(pixelValue)
        cv2.imshow("masked", maskMatrix)
        
        maskMatrix = np.array(maskMatrix, dtype=np.uint8)
        
        image, contours, hierarchy = cv2.findContours(maskMatrix.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        largestArea = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if(largestArea < peri):
                screenCnt = approx
                largestArea = peri

        x,y,w,h = cv2.boundingRect(screenCnt)
        
        topLeft = (x,y)
        bottomLeft = (x,y+h)
        bottomRight = (x+w,y+h)
        topRight = (x+w,y)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        boundingBoxArea = w*h
        boxDistance = m * boundingBoxArea + b
        
        json.dump("ImageNo - ", testingOutputFolder)
        json.dump(filename, testingOutputFolder)
        json.dump(", TopLeftX - ", testingOutputFolder)
        json.dump(x, testingOutputFolder)
        json.dump(", TopLeftY - ", testingOutputFolder)
        json.dump(y, testingOutputFolder)
        json.dump(", BottomRightX - ", testingOutputFolder)
        json.dump(x+w, testingOutputFolder)
        json.dump(", BottomRightY - ", testingOutputFolder)
        json.dump(y+h, testingOutputFolder)
        json.dump(", Distance - ", testingOutputFolder)
        json.dump(boxDistance, testingOutputFolder)
        testingOutputFolder.write("\n")
               
        cv2.imshow('image',img)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    testingOutputFolder.close()

# Function calls to evaluate test images. PixelDataGenerator, PrintMeanData, PrintPopulationVarianceData,
# CovarianceCalculator, and PrintCovarianceData are unused to speed up runtime and are only used when retraining.

# PixelDataGenerator()
MeanPixelDataCalculator()
#PrintMeanData()
PopulationVarianceCalculator()
#PrintPopulationVarianceData()
#CovarianceCalculator()
#PrintCovarianceData()
BarrelLinearRegression()
BarrelDetection()
