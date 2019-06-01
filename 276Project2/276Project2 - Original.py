from __future__ import division
import pickle
import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import math
import transforms3d

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')  # need for python 3
    return d

def QuaternionMagnitude(quaternion):
    resultQuaternion = [0] * len(quaternion)
    for i in range(0, len(quaternion)):
        resultQuaternion[i] = quaternion[i]**2
    resultNorm = np.sum(resultQuaternion)
    return resultNorm

def QuaternionNorm(quaternion):
    resultMagnitude = math.sqrt(QuaternionMagnitude(quaternion))
    return resultMagnitude

def QuaternionConjugate(quaternion):
    resultQuaternion = [0,0,0,0]
    resultQuaternion[0] = quaternion[0]
    resultQuaternion[1] = -quaternion[1]
    resultQuaternion[2] = -quaternion[2]
    resultQuaternion[3] = -quaternion[3]
    return resultQuaternion

def QuaternionMultiply(firstQuaternion, secondQuaternion):
    resultQuaternion = [0,0,0,0]
    resultQuaternion[0] = (firstQuaternion[0]*secondQuaternion[0] - firstQuaternion[1]*secondQuaternion[1] - firstQuaternion[2]*secondQuaternion[2] - firstQuaternion[3]*secondQuaternion[3])
    resultQuaternion[1] = (firstQuaternion[0]*secondQuaternion[1] + firstQuaternion[1]*secondQuaternion[0] + firstQuaternion[2]*secondQuaternion[3] - firstQuaternion[3]*secondQuaternion[2])
    resultQuaternion[2] = (firstQuaternion[0]*secondQuaternion[2] - firstQuaternion[1]*secondQuaternion[3] + firstQuaternion[2]*secondQuaternion[0] + firstQuaternion[3]*secondQuaternion[1])
    resultQuaternion[3] = (firstQuaternion[0]*secondQuaternion[3] + firstQuaternion[1]*secondQuaternion[2] - firstQuaternion[2]*secondQuaternion[1] + firstQuaternion[3]*secondQuaternion[0])
    return resultQuaternion

def QuaternionLog(quaternion):
    resultQuaternion = [0,0,0,0]
    fullQuaternionNorm = QuaternionNorm(quaternion)
    vectorQuaternionNorm = QuaternionNorm(quaternion[1:4])
    
    if(fullQuaternionNorm == 0):
        resultQuaternion[0] = 0
    else:
        resultQuaternion[0] = math.log(fullQuaternionNorm)
    if(quaternion[1] == 0):
        resultQuaternion[1] = 0
    else:
        resultQuaternion[1] = (quaternion[1]/vectorQuaternionNorm)*math.acos(quaternion[0]/fullQuaternionNorm)
        
    if(quaternion[2] == 0):
        resultQuaternion[2] = 0
    else:
        resultQuaternion[2] = (quaternion[2]/vectorQuaternionNorm)*math.acos(quaternion[0]/fullQuaternionNorm)
        
    if(quaternion[3] == 0):
        resultQuaternion[3] = 0
    else:
        resultQuaternion[3] = (quaternion[3]/vectorQuaternionNorm)*math.acos(quaternion[0]/fullQuaternionNorm)
    return resultQuaternion

def QuaternionExp(quaternion):
    resultQuaternion = [0,0,0,0]
    vectorQuaternionNorm = QuaternionNorm(quaternion[1:4])
    exponentValue = math.exp(quaternion[0])
    
    resultQuaternion[0] = exponentValue * math.cos(vectorQuaternionNorm)
    if(quaternion[1] == 0):
        resultQuaternion[1] = 0
    else:
        resultQuaternion[1] = exponentValue*((quaternion[1]/vectorQuaternionNorm)*math.sin(vectorQuaternionNorm))

    if(quaternion[2] == 0):
        resultQuaternion[2] = 0
    else:
        resultQuaternion[2] = exponentValue*((quaternion[2]/vectorQuaternionNorm)*math.sin(vectorQuaternionNorm))
        
    if(quaternion[3] == 0):
        resultQuaternion[3] = 0
    else:
        resultQuaternion[3] = exponentValue*((quaternion[3]/vectorQuaternionNorm)*math.sin(vectorQuaternionNorm))
    return resultQuaternion

def QuaternionRotation(rotation,quaternion):
    tempResult = QuaternionMultiply(rotation, quaternion)
    rotatedQuaternion = QuaternionMultiply(tempResult, QuaternionConjugate(rotation)) 
    return rotatedQuaternion

def QuaternionInverse(quaternion):
    resultQuaternion = [0,0,0,0]
    quaternionConjugate = QuaternionConjugate(quaternion)
    quaternionNormSquared = QuaternionNorm(quaternion)**2
    
    resultQuaternion[0] = quaternionConjugate[0]/quaternionNormSquared
    resultQuaternion[1] = quaternionConjugate[1]/quaternionNormSquared
    resultQuaternion[2] = quaternionConjugate[2]/quaternionNormSquared
    resultQuaternion[3] = quaternionConjugate[3]/quaternionNormSquared
    
    return resultQuaternion

def CalibrateReadings():
    biasWX = biasWY = biasWZ = biasACCX = biasACCY = biasACCZ = 0
    biasRange = 100
    
    for i in range(0,biasRange):
        biasWX += imud["vals"][0][i]
        biasWY += imud["vals"][1][i]
        biasWZ += imud["vals"][2][i]
        biasACCX += imud["vals"][4][i]
        biasACCY += imud["vals"][5][i]
        biasACCZ += (imud["vals"][3][i] - 1)
    biasWX = biasWX/biasRange
    biasWY = biasWY/biasRange
    biasWZ = biasWZ/biasRange
    biasACCX = biasACCX/biasRange
    biasACCY = biasACCY/biasRange
    biasACCZ = biasACCZ/biasRange
    print(biasWX)
    print(biasWY)
    print(biasWZ)
    print(biasACCX)
    print(biasACCY)
    print(biasACCZ)
    
    for i in range(0, len(imud["ts"][0])):
        newImud[0][i] = -(imud["vals"][0][i] - biasWX)*(3300/(1023*300))*(np.pi/180.0)
        newImud[1][i] = -(imud["vals"][1][i] - biasWY)*(3300/(1023*300))*(np.pi/180.0)
        newImud[2][i] = (imud["vals"][2][i] - biasWZ)*(3300/(1023*300))*(np.pi/180.0)
        newImud[4][i] = (imud["vals"][4][i] - biasACCX)*(3300/(1023*3.3))*(np.pi/180.0)
        newImud[5][i] = (imud["vals"][5][i] - biasACCY)*(3300/(1023*3.3))*(np.pi/180.0)
        newImud[3][i] = (imud["vals"][3][i] - biasACCZ)*(3300/(1023*3.3))*(np.pi/180.0)
    print(newImud)

def PlotGraphs():
    vicdRoll = np.empty(len(vicd["ts"][0]))
    vicdPitch = np.empty(len(vicd["ts"][0]))
    vicdYaw = np.empty(len(vicd["ts"][0]))
    
    imudRoll = np.empty(len(vicd["ts"][0]))
    imudPitch = np.empty(len(vicd["ts"][0]))
    imudYaw = np.empty(len(vicd["ts"][0]))
    
    currentQuaternion = [1,0,0,0]
    
    for i in range(1, len(vicd["ts"][0])):
        rotMatrix = np.empty((3,3))
        # X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3
        for j in range(0,3):
            for k in range(0,3):
                rotMatrix[j][k] = float(vicd["rots"][j][k][i])
        eulerAngles = transforms3d.euler.mat2euler(rotMatrix)
        
        vicdRoll[i] = eulerAngles[0]
        vicdPitch[i] = eulerAngles[1]
        vicdYaw[i] = eulerAngles[2]

        wT = (newImud[4][i],newImud[5][i],newImud[3][i])
        deltaT = imud["ts"][0][i] - imud["ts"][0][i - 1]
        quaternionExponentValue = [0,(wT[0]*deltaT)/2,(wT[1]*deltaT)/2,(wT[2]*deltaT)/2]
        tempQuaternion = QuaternionMultiply(currentQuaternion, QuaternionExp(quaternionExponentValue))
        eulerAngles = transforms3d.euler.quat2euler(tempQuaternion)
        
        imudRoll[i] = eulerAngles[0]
        imudPitch[i] = eulerAngles[1]
        imudYaw[i] = eulerAngles[2]
        
        currentQuaternion = tempQuaternion

    plt.plot(vicd["ts"][0],imudRoll,color = 'red')
    plt.plot(vicd["ts"][0],vicdRoll,color = 'blue')
    plt.show()
    plt.plot(vicd["ts"][0],imudPitch,color = 'red')
    plt.plot(vicd["ts"][0],vicdPitch,color = 'blue')
    plt.show()
    plt.plot(vicd["ts"][0],imudYaw,color = 'red')
    plt.plot(vicd["ts"][0],vicdYaw,color = 'blue')
    plt.show()

def QuaternionAverage(quaternionValues):
    global eVI
    alpha0 = 0
    alphaI = (1/(2*3))
    qe = [[0,0,0,0] for i in range(7)]
    rotVectors = [[0,0,0,0] for i in range(7)]
    qt = [0, 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)]
    eVSum = [0,0,0]
    updatedEVSum = [0,0,0,0]
    epsilonValue = .0001
    i = 0
    
    while(1):
        for i in range(0,len(quaternionValues)):
#             print("QT - ")
#             print(qt)
            qe[i] = QuaternionMultiply(QuaternionInverse(qt),quaternionValues[i])
#             print("QE - ")
#             print(qe[i])
            
    #         eularAngles = transforms3d.euler.quat2euler(qe[i])
    #         rotVectors[i][0] = 0
    #         rotVectors[i][1] = eularAngles[0]
    #         rotVectors[i][2] = eularAngles[1]
    #         rotVectors[i][3] = eularAngles[2]
            quaternionLogMap = QuaternionLog(qe[i])
#             print("LOG MAP - ")
#             print(quaternionLogMap)
            rotVectors[i][0] = 0.0
            rotVectors[i][1] = 2*quaternionLogMap[1]
            rotVectors[i][2] = 2*quaternionLogMap[2]
            rotVectors[i][3] = 2*quaternionLogMap[3]
            eVI[i][0] = rotVectors[i][1]
            eVI[i][1] = rotVectors[i][2]
            eVI[i][2] = rotVectors[i][3]
#             print("ROTATION VECTOR - ")
#             print(rotVectors[i])
             
            eVINorm = QuaternionNorm(eVI[i])
#             print("EVI NORM - ")
#             print(eVINorm)
            if(eVINorm == 0):
                eVI[i][0] = 0
                eVI[i][1] = 0
                eVI[i][2] = 0
            else:
                eVI[i][0] = (-math.pi+divmod(eVINorm+math.pi,2*math.pi)[1])*(eVI[i][0]/eVINorm)
                eVI[i][1] = (-math.pi+divmod(eVINorm+math.pi,2*math.pi)[1])*(eVI[i][1]/eVINorm)
                eVI[i][2] = (-math.pi+divmod(eVINorm+math.pi,2*math.pi)[1])*(eVI[i][2]/eVINorm)
#             print("RESTRICTED EVI - ")
#             print(eVI[i])
            
#             print""
        
        for i in range(0,len(quaternionValues)):
            if(i == 0):
                eVSum[0] += alpha0*eVI[i][0]
                eVSum[1] += alpha0*eVI[i][1]
                eVSum[2] += alpha0*eVI[i][2]
            else:
                eVSum[0] += alphaI*eVI[i][0]
                eVSum[1] += alphaI*eVI[i][1]
                eVSum[2] += alphaI*eVI[i][2]
#             print("EVSUM - ")
#             print(eVSum)
        updatedEVSum[0] = 0.0
        updatedEVSum[1] = eVSum[0]/2
        updatedEVSum[2] = eVSum[1]/2
        updatedEVSum[3] = eVSum[2]/2
#         print("UPDATED EV SUM - ")
#         print(updatedEVSum)
#         print("QT - ")
#         print(qt)
#         print("UPDATED EV SUM EXPONENTIAL - ")
#         print(QuaternionExp(updatedEVSum))

        updatedQT = QuaternionMultiply(qt,QuaternionExp(updatedEVSum))
#         print("UPDATED QT - ")
#         print(updatedQT)
        
        eVNorm = QuaternionNorm(eVSum)
#         print("EV NORM - ")
#         print(eVNorm)
        
        if(eVNorm < epsilonValue):
#             print("")
#             print("QUATERNION RESULT - ")
#             print(updatedQT)
#             print("EULER RESULT - ")
#             print(transforms3d.euler.quat2euler(updatedQT))
#             print""
#             print""
            return updatedQT

        qt = updatedQT
        eVSum[0] = 0
        eVSum[1] = 0
        eVSum[2] = 0
#         print""
#         print""
        
#         print""
#         print("--------------NEW LOOP---------------")
#         print""
        
        
def UnsentedKalmanFilter():
    global eVI
    sigmaPointNumber = 7
    qt = [1,0,0,0]
    Pt = .001 * np.identity(3)
    Q = .001 * np.identity(3)
    PR = .001 * np.identity(3)
    alpha0 = (1/2)
    alphaI = (1/(4*3))
    qt1Giventi = [[0,0,0,0] for i in range(0,sigmaPointNumber)]
    Zt1i = [[0,0,0] for i in range(0,sigmaPointNumber)]
    Zt1Mean = [0,0,0]
    PZZ = [[0,0,0],[0,0,0],[0,0,0]]
    Eti = [[0,0,0],[0,0,0],[0,0,0]]
    qtResults = [[0,0,0,0] for i in range(0,len(vicd["ts"][0]))]
#     eI = [[0,0,0,0] for i in range(6)]
    
    for j in range(1, len(vicd["ts"][0])):
        
        print""
        print("PREDICT")
        print""
        # PREDICTION STEPS 1-6
        # -------------------- STEP 1 --------------------
        wt = [newImud[4][j],newImud[5][j],newImud[3][j]]
        z = [newImud[0][j],newImud[1][j],newImud[2][j]]
        deltaT = imud["ts"][0][j] - imud["ts"][0][j - 1]
        print("WT DATA - ")
        print(wt)
        print("Z DATA - ")
        print(z)
        print("DELTA T - ")
        print(deltaT)
        
        # -------------------- STEP 2 --------------------
        Et0 = [0,0,0]
        Eti = np.linalg.cholesky(np.sqrt(3)*(Pt + Q))
        print("ETI - ")
        print(Eti)
        
        # -------------------- STEP 3 --------------------
        IMUexponentValue = QuaternionExp([0,(wt[0]*deltaT)/2,(wt[1]*deltaT)/2,(wt[2]*deltaT)/2])
        print(IMUexponentValue)
        
        EtExponentValue0 = QuaternionExp([0,Et0[0]/2,Et0[1]/2,Et0[2]/2])
        print(EtExponentValue0)
        qt1Giventi[0] = QuaternionMultiply(QuaternionMultiply(qt,EtExponentValue0),IMUexponentValue)
        print("QT1GIVENT0 - ")
        print(qt1Giventi[0])
        
        print("QT1GIVENTI - ")
        for i in range(1,sigmaPointNumber):
            if(i < 4):
                signValue = 1
#                 print("ETI * SIGN VALUE - ")
#                 print(signValue * Eti[0][i - 1])
#                 print(signValue * Eti[1][i - 1])
#                 print(signValue * Eti[2][i - 1])
                EtExponentValuei = QuaternionExp([0,(signValue*Eti[0][i - 1])/2,(signValue*Eti[1][i - 1])/2,(signValue*Eti[2][i - 1])/2])
            else:
                signValue = -1
#                 print("ETI * SIGN VALUE - ")
#                 print(signValue * Eti[0][i - 4])
#                 print(signValue * Eti[1][i - 4])
#                 print(signValue * Eti[2][i - 4])
                EtExponentValuei = QuaternionExp([0,(signValue*Eti[0][i - 4])/2,(signValue*Eti[1][i - 4])/2,(signValue*Eti[2][i - 4])/2])
                
    #         print(EtExponentValuei)
            qt1Giventi[i] = QuaternionMultiply(QuaternionMultiply(qt,EtExponentValuei),IMUexponentValue)
            print(qt1Giventi[i])
    #     print(qt1Giventi)
        
        # -------------------- STEP 4 --------------------
        print("QUATERNION AVERAGE - ")
        qt1GiventMean = QuaternionAverage(qt1Giventi)
        print(qt1GiventMean)
        print("EULER RESULT - ")
        print(transforms3d.euler.quat2euler(qt1GiventMean))
        print("eI - ")
        print(eVI)
            
        # -------------------- STEP 6 --------------------
        eVI0 = np.array([eVI[0]])
        Pt1Givent  = (0)*np.multiply(eVI0,np.transpose(eVI0))
        for i in range(1,sigmaPointNumber):
            eVII = np.array([eVI[i]])
            Pt1Givent = Pt1Givent + (1/(2*3))*np.multiply(eVII,np.transpose(eVII))
        print("P t1|t")
        print(Pt1Givent)
        
        
        #PLOT QT1GIVENTMEAN HERE AGAINST VIC DATA
        qtResults[j] = qt1GiventMean
        qt = qt1GiventMean
        Pt = Pt1Givent
        
        print""
        print("UPDATE")
        print""
        # UPDATE STEPS 1-6
        # -------------------- STEP 1 --------------------
        print("Z t+1 i - ")
        gravityValue = [0,0,0,1]
        for i in range(0,sigmaPointNumber):
            tempZt1Value = QuaternionMultiply(QuaternionMultiply(QuaternionConjugate(qt1Giventi[i]), gravityValue),qt1Giventi[i])
    #         print(tempZt1Value)
            Zt1i[i][0] = tempZt1Value[1]
            Zt1i[i][1] = tempZt1Value[2]
            Zt1i[i][2] = tempZt1Value[3]
        print(Zt1i)
        print(Zt1i[1][0])
      
        # -------------------- STEP 2 --------------------
        Zt1Mean[0] = (2)*Zt1i[0][0]
        Zt1Mean[1] = (2)*Zt1i[0][1]
        Zt1Mean[2] = (2)*Zt1i[0][2]
        for i in range(1,sigmaPointNumber):
            Zt1Mean[0] += (1/(2*3))*Zt1i[i][0]
            Zt1Mean[1] += (1/(2*3))*Zt1i[i][1]
            Zt1Mean[2] += (1/(2*3))*Zt1i[i][2]
        print("Z t+1 Mean - ")
        print(Zt1Mean)
          
        # -------------------- STEP 3 --------------------
        tempResultA = np.subtract(Zt1Mean, Zt1i[0])
        tempResultA = np.array([tempResultA])
        tempResultB = tempResultA.T
        PZZ = (2)*np.multiply(tempResultA,tempResultB)
    #     print("A - ")
    #     print(tempResultA)
    #     print("B - ")
    #     print(tempResultB)
    #     print("C - ")
    #     print(PZZ)
        PZZ = 0
           
        for i in range(1,sigmaPointNumber):
            tempResultA = np.subtract(Zt1i[i], Zt1Mean)
            tempResultA = np.array([tempResultA])
            tempResultB = tempResultA.T
            PZZ = PZZ + (1/(2*3))*np.multiply(tempResultA,tempResultB)
    #         print("CURRENT VALUES TO ADD - ")
    #         print((1/(4*3))*np.multiply(tempResultA,tempResultB))
    #         print("UPDATED PZZ - ")
    #         print(PZZ)
        print("FINAL PZZ - ")
        print(PZZ)
          
        # -------------------- STEP 4 --------------------
        PVV = PZZ + PR
        print("PVV - ")
        print(PVV)
          
        # -------------------- STEP 5 --------------------
        tempResultA = np.subtract(Zt1i[0],Zt1Mean)
        tempResultA = np.array([tempResultA])
        tempResultB = tempResultA.T
        PXZ = 2*np.multiply(eVI0,tempResultB)
    #     print("A - ")
    #     print(tempResultA)
    #     print("B - ")
    #     print(tempResultB)
    #     print("C - ")
    #     print(tempResultC)
    #     print("INITIAL PXZ - ")
    #     print(PXZ)
          
        for i in range(1,sigmaPointNumber):
            tempResultA = np.subtract(Zt1i[i],Zt1Mean)
            tempResultA = np.array([tempResultA])
            tempResultB = tempResultA.T
            PXZ = PXZ + (1/(2*3))*np.multiply(eVI0,tempResultB)
    #         print("CURRENT VALUES TO ADD - ")
    #         print(np.multiply(eVI0,tempResultB))
    #         print("UPDATED PXZ - ")
    #         print(PXZ)
      
        print("FINAL PXZ - ")        
        print(PXZ)
          
        # -------------------- STEP 6 --------------------
        Kt1 = PXZ + np.linalg.inv(PVV)
        print("Kt1 - ")        
        print(Kt1)
          
          
        print""
        print("NEXT ITERATION q t+1|t+1 and P t+1|t+1")
        print""
        # -------------------- STEP 1 --------------------
        tempResultA = np.subtract(z,Zt1Mean)
        tempResultB = np.dot(Kt1,tempResultA)
        tempResultC = np.divide(tempResultB,2)
        tempResuldD = QuaternionExp([0,tempResultC[0],tempResultC[1],tempResultC[2]])
        qt = QuaternionMultiply(qt1GiventMean,tempResuldD)
        print("A - ")
        print(tempResultA)
        print("B - ")
        print(tempResultB)
        print("C - ")
        print(tempResultC)
        print("D - ")
        print(tempResuldD)
        print("NEW QT - ")
        print(qt)
          
        qtResults[j] = qt
          
        # -------------------- STEP 2 --------------------
        print("PT VALUES - ")
        print(Pt1Givent)
        print(Kt1)
        print(PVV)
        print(Kt1.T)
        Pt = np.subtract(Pt1Givent,np.dot(np.dot(Kt1,PVV),Kt1.T))
        print""
        print(np.dot(Kt1,PVV))
        print(np.dot(np.dot(Kt1,PVV),Kt1.T))
        print("NEW PT - ")
        print(Pt)
          
        Pt = np.absolute(Pt)
        print(Pt)
          
        print("REPEATING")
        print(j)
        
    return qtResults
    
def PlotUKFResults(qtResults):
    resultsRoll = np.empty(len(vicd["ts"][0]))
    resultsPitch = np.empty(len(vicd["ts"][0]))
    resultsYaw = np.empty(len(vicd["ts"][0]))
    
    for i in range(1, len(vicd["ts"][0])):
        eulerAngles = transforms3d.euler.quat2euler(qtResults[i])
        
        resultsRoll[i] = eulerAngles[0]
        resultsPitch[i] = eulerAngles[1]
        resultsYaw[i] = eulerAngles[2]

    plt.plot(vicd["ts"][0],resultsRoll,color = 'red')
    plt.show()
    plt.plot(vicd["ts"][0],resultsPitch,color = 'red')
    plt.show()
    plt.plot(vicd["ts"][0],resultsYaw,color = 'red')
    plt.show()

dataset="1"
# cfile = "trainset/cam/cam" + dataset + ".p"
ifile = "trainset/imu/imuRaw" + dataset + ".p"
vfile = "trainset/vicon/viconRot" + dataset + ".p"

ts = tic()
# camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

newImud = np.empty((6,len(imud["ts"][0])))
eVI = [[0,0,0] for i in range(7)]
	
# quaternionValues = [[0,0,0,0] for i in range(7)]

# quaternionValues[0] = [0.08715561, 0.996,0,0]
# quaternionValues[1] = [ 0.636,-0.772,0,0]
# quaternionValues[2] = [-0.707,0.707,0,0]

# quaternionValues[0] = [1,1,1,1]
# quaternionValues[1] = [1,1,1,1]
# quaternionValues[2] = [1,1,1,1]
# quaternionValues[3] = [1,1,1,1]
# quaternionValues[4] = [1,1,1,1]
# quaternionValues[5] = [1,1,1,1]
# quaternionValues[6] = [1,1,1,1]
# 
# print(quaternionValues)
# print(QuaternionAverage(quaternionValues))

CalibrateReadings()
PlotGraphs()
# qtResults = UnsentedKalmanFilter()
# PlotUKFResults(qtResults)
