# -*- coding: utf-8 -*-
#! python3

import vrep
import numpy as np
import time
import csv
from PIL import Image
import array
import random
import math
import os
import sys
import requests
import io
import matplotlib.pyplot as plt

def moveToDummy(dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold):
    inInts = [dummy, base, maxVel, maxAccel, maxJerk, ikSteps]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, "Baxter_rightArm_joint1", vrep.sim_scripttype_childscript, 'pyMoveToDummy', inInts, [], [], emptyBuff, vrep.simx_opmode_blocking)
    # Wait until the end of the movement:
    runningPath = True
    startMove = time.clock()
    while runningPath:
        runningPath = isIkRunning(hand, dummy, base, pThreshold, oThreshold)
        if time.clock()-startMove > 4:
            return
            # Stop simulation:
            #vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
            # Now close the connection to V-REP:
            #vrep.simxFinish(clientID)
            #sys.exit("IK error, exit")

def moveToConfig(config, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold):
    inInts = [maxVel, maxAccel, maxJerk, fkSteps]
    res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, "Baxter_rightArm_joint1", vrep.sim_scripttype_childscript, 'pyMoveToConfig', inInts, config, [], emptyBuff, vrep.simx_opmode_blocking)    
    # Wait until the end of the movement:
    runningPath=True
    startMove = time.clock()
    while runningPath:
        runningPath = isFkRunning(config, oThreshold, emptyBuff)
        if time.clock()-startMove > 30:
            # Stop simulation:
            #vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
            # Now close the connection to V-REP:
            #vrep.simxFinish(clientID)
            #sys.exit("FK error, exit")
            return

def createDummy(handle, base):
    inInts = [handle, base]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, "Baxter_rightArm_joint1", vrep.sim_scripttype_childscript, 'pyCreateDummy', inInts, [], [], emptyBuff, vrep.simx_opmode_blocking)
    if len(retInts) != 1:
        # Stop simulation:
        vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
        # Now close the connection to V-REP:
        vrep.simxFinish(clientID)
        sys.exit("create dummy error, exit")
    return retInts[0]


def getImg(camera, length, width):
    res, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 0, vrep.simx_opmode_blocking)
    image_byte_array = array.array('b',image)
    img = Image.frombuffer("RGB", (length,width), bytes(image_byte_array), "raw", "RGB", 0, 1)
    #img.save(path + str(pid) + '.jpg')
    box = (188, 148, 412, 372)

    img1 = img.crop(box)
    # plt.figure("Image")
    # plt.imshow(img1)
    # plt.show()
    return img1

def getBuffer(length, width, perspAngle, camera, hand, graspHeight, gripperLength):
    res, handPosition = vrep.simxGetObjectPosition(clientID, hand, hand, vrep.simx_opmode_blocking)
    res, cameraPosition = vrep.simxGetObjectPosition(clientID, camera, hand, vrep.simx_opmode_blocking)
    c = (length/2) / math.sin(perspAngle/2)
    b = math.sqrt(np.square(width/2)+np.square(length/2))
    h = math.sqrt(np.square(c)-np.square(b))
    propPara = h / (cameraPosition[2]-handPosition[2]+graspHeight+gripperLength)
    bufferX = (handPosition[0]-cameraPosition[0]) * propPara
    bufferY = (handPosition[1]-cameraPosition[1]) * propPara
    bufferX = length/2 + bufferX
    bufferY = width/2 + bufferY
    return int(bufferX), int(bufferY)


def goForRotate(camera, dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold):
    res,retInts,configBeforeRotate,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGetConfig', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    configAfterRotate = list(configBeforeRotate)
    #get rotate angle
    #angle = getRotateAngle(image)
    # configAfterRotate[6] = random.uniform(-math.pi, math.pi)
    url ='http://172.18.160.172:8080/inference'
    img = getImg(camera, 640, 400)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    image = {'image': imgByteArr}
    res = requests.post(url, files=image)
    configAfterRotate[6] = math.radians(int(res.text))
    # print('', res.text)

   
    moveToConfig(configAfterRotate, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold)
    res, handOrientation = vrep.simxGetObjectOrientation(clientID, hand, base, vrep.simx_opmode_blocking)
    return handOrientation[2], configBeforeRotate

def goForRotate0(dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold):
    res,retInts,configBeforeRotate,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGetConfig', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    configAfterRotate = list(configBeforeRotate)
    # get rotate angle
    #angle = getRotateAngle(image)
    #configAfterRotate[6] = random.uniform(-math.pi, math.pi)
    configAfterRotate[6] = 0.0
    moveToConfig(configAfterRotate, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold)

def goForRandomRotate(dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold):
    res,retInts,configBeforeRotate,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGetConfig', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    configAfterRotate = list(configBeforeRotate)
    configAfterRotate[6] = random.uniform(-math.pi, math.pi)
    moveToConfig(configAfterRotate, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold)
    res, handOrientation = vrep.simxGetObjectOrientation(clientID, hand, base, vrep.simx_opmode_blocking)
    return handOrientation[2], configBeforeRotate

def goForOffset(dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold, offsetX, offsetY):
    randomX = random.uniform(-offsetX, offsetX) / 2
    randomY = random.uniform(-offsetY, offsetY) / 2
    res, dummyPosition = vrep.simxGetObjectPosition(clientID, dummy, base, vrep.simx_opmode_blocking)
    dummyPosition = [dummyPosition[0]+randomX, dummyPosition[1]+randomY, dummyPosition[2]]
    vrep.simxSetObjectPosition(clientID, dummy, base, dummyPosition, vrep.simx_opmode_blocking)
    moveToDummy(dummy, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)

def samePoint(p1, p2, threshold):
    for i in range(3):
        if abs(p1[i] - p2[i]) >= threshold:
            return False
    return True
    
def isIkRunning(handle, dummy, base, pThreshold, oThreshold):
    res, handlePosition = vrep.simxGetObjectPosition(clientID, handle, base, vrep.simx_opmode_blocking)
    res, dummyPosition = vrep.simxGetObjectPosition(clientID, dummy, base, vrep.simx_opmode_blocking)
    res, handleOrientation = vrep.simxGetObjectOrientation(clientID, handle, base, vrep.simx_opmode_blocking)
    res, dummyOrientation = vrep.simxGetObjectOrientation(clientID, dummy, base, vrep.simx_opmode_blocking)
    if samePoint(handlePosition, dummyPosition, pThreshold) and samePoint(handleOrientation, dummyOrientation, oThreshold):
        return False
    else:
        return True

def isFkRunning(goal, othreshold, emptyBuff):
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, "Baxter_rightArm_joint1", vrep.sim_scripttype_childscript, 'pyGetConfig', [], [], [], emptyBuff, vrep.simx_opmode_blocking)
    config = retFloats
    for i in range(len(config)):
        if abs(config[i] - goal[i]) >= othreshold:
            return True
    return False

def goAndGrasp(startP, camera, pid, obj, graspHeight, gripperLength, offsetX, offsetY, regionY, regionZ, box, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold):
    # get object position
    res, objectPosition = vrep.simxGetObjectPosition(clientID, obj, base, vrep.simx_opmode_blocking)
    if isValid(obj, box, regionY, regionZ) == False:
        return 1, 0, 0, 0, 0, 0, 0
    # go and grasp
    # create point above object
    aboveP1 = createDummy(startP, base)
    #print('aboveP1 is ', aboveP1)
    res, position = vrep.simxGetObjectPosition(clientID, aboveP1, base, vrep.simx_opmode_blocking)
    abovePPosition = np.array([objectPosition[0], objectPosition[1], position[2]])
    vrep.simxSetObjectPosition(clientID, aboveP1, base, abovePPosition, vrep.simx_opmode_blocking)
    # gripper open
    inInts = [motorForce]
    inFloats = [motorVelocity1, motorVelocity2]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGripper', inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(0.5)
    # go to aboveP
    moveToDummy(aboveP1, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
    # offset
    # goForOffset(aboveP1, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold, offsetX, offsetY)
    aboveP2 = createDummy(hand, base)
    #print('another aboveP2 is ', aboveP2)
    time.sleep(0.5)
    #save img
    #saveImg(camera, pid, 640, 400, 'C:/Users/zhang/Desktop/picturesRight'+str(proId)+'/'+str(testId)+'/objr')
    goForRotate0(aboveP2, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold)
    #saveImg(camera, pid, 640, 400, 'C:/Users/zhang/Desktop/dataSet/'+str(pid)+'objr')
    bufferX, bufferY = getBuffer(640, 400, math.pi/3, camera, hand, graspHeight, gripperLength)
    # rotate
    # graspAngle=?
    angle, configBeforeRotate = goForRotate(camera, aboveP2, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold)
    aboveP3 = createDummy(hand, base)

    # create grasp point
    graspP = createDummy(aboveP3, base)
    #print('graspP is ', graspP)
    res, position = vrep.simxGetObjectPosition(clientID, graspP, base, vrep.simx_opmode_blocking)
    graspPPosition = np.array([position[0], position[1], objectPosition[2]+gripperLength - 0.04])
    vrep.simxSetObjectPosition(clientID, graspP, base, graspPPosition, vrep.simx_opmode_blocking)
    # down to graspP
    moveToDummy(graspP, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
    # gripper close
    inInts = [motorForce]
    inFloats = [-motorVelocity1, -motorVelocity2]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGripper', inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(1.5)
    # up to aboveP
    moveToDummy(aboveP3, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
     
    # remove dummies
    res=vrep.simxRemoveObject(clientID,aboveP1,vrep.simx_opmode_blocking)
    res=vrep.simxRemoveObject(clientID,aboveP2,vrep.simx_opmode_blocking)
    res=vrep.simxRemoveObject(clientID,graspP,vrep.simx_opmode_blocking)
    return 0, graspPPosition, angle, configBeforeRotate, aboveP3, bufferX, bufferY

def goAndPutDown(startP, regionY, regionZ, box, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold):
    # create point above final put down position randomly
    aboveFinalP1 = createDummy(startP, box)
    #randomY = random.uniform(regionY[0], regionY[1])
    #randomZ = random.uniform(regionZ[0], regionZ[1])
    #res, aboveFinalPPosition = vrep.simxGetObjectPosition(clientID, aboveFinalP, box, vrep.simx_opmode_blocking)
    #aboveFinalPPosition = [aboveFinalPPosition[0], randomY, randomZ]
    #vrep.simxSetObjectPosition(clientID, aboveFinalP, box, aboveFinalPPosition, vrep.simx_opmode_blocking)
    # go to aboveFinalP
    #moveToDummy(aboveFinalP, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
    # rotate
    angle, configBeforeRotate = goForRandomRotate(aboveFinalP1, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, oThreshold)
    aboveFinalP2 = createDummy(hand, base)
    # create put down point
    putDownP1 = createDummy(aboveFinalP2, -1)
    res, putDownPPosition = vrep.simxGetObjectPosition(clientID, putDownP1, -1, vrep.simx_opmode_blocking)
    putDownPPosition = [putDownPPosition[0], putDownPPosition[1], tableHeight + gripperLength + 0.01]
    vrep.simxSetObjectPosition(clientID, putDownP1, -1, putDownPPosition, vrep.simx_opmode_blocking)
    # down to putDownP
    putDownP2 = createDummy(putDownP1,base)
    moveToDummy(putDownP2, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
    # open
    inInts = [motorForce]
    inFloats = [motorVelocity1, motorVelocity2]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGripper', inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(1.5)
    # up to aboveFinal
    moveToDummy(aboveFinalP2, hand, base, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
    #return configBeforeRotate
    res=vrep.simxRemoveObject(clientID,aboveFinalP1,vrep.simx_opmode_blocking)
    res=vrep.simxRemoveObject(clientID,aboveFinalP2,vrep.simx_opmode_blocking)
    res=vrep.simxRemoveObject(clientID,putDownP1,vrep.simx_opmode_blocking)
    res=vrep.simxRemoveObject(clientID,putDownP2,vrep.simx_opmode_blocking)

def isGrasp(obj, tableHeight):
    res, position = vrep.simxGetObjectPosition(clientID, obj, -1, vrep.simx_opmode_blocking)
    if position[2] > tableHeight + 0.15:
        return True
    else:
        return False

def isValid(obj, box, regionY, regionZ):
    res, objPosition = vrep.simxGetObjectPosition(clientID, obj, box, vrep.simx_opmode_blocking)
    if (objPosition[1] < regionY[1] and objPosition[1] > regionY[0]) and (objPosition[2] < regionZ[1] and objPosition[2] > regionZ[0]):
        return True
    else:
        return False

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1',19998,True,True,5000,5) # Connect to V-REP
if clientID != -1:
    print ('Connected to remote API server')
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
        
    # joint parameters
    maxVel=3
    maxAccel=1
    maxJerk=8000
    ikSteps=200
    fkSteps=160
    emptyBuff=bytearray()
        
    # basic parameters
    objectNum=6
    tableHeight=0.9
    graspHeight=0.1563
    gripperLength=0.15
        
    pThreshold=0.01
    oThreshold=0.01
    #regionY=[-0.1875,0.1875]
    #regionZ=[-0.2375,0.2375]
    regionY=[-0.219,0.219]
    regionZ=[-0.279,0.279]
    offsetX=0.04
    offsetY=0.04
        
    res, baxterBaseHandle = vrep.simxGetObjectHandle(clientID, 'Baxter', vrep.simx_opmode_blocking)
    res, baxterRightHand = vrep.simxGetObjectHandle(clientID, 'Baxter_rightArm_connector', vrep.simx_opmode_blocking)
    res, rightBox = vrep.simxGetObjectHandle(clientID, 'box_right', vrep.simx_opmode_blocking)
    res, rightCamera = vrep.simxGetObjectHandle(clientID, 'Vision_sensor_right', vrep.simx_opmode_blocking)
        
    # gripper parameters
    motorVelocity1=-0.1 # m/s
    motorVelocity2=-0.1 # m/s
    motorForce=40 # N
        
    #testId = sys.argv[1]
    #proId = sys.argv[2]
    #print("loop "+str(testId)+" running")
    print("loop is running")
    #os.makedirs("C:/Users/zhang/Desktop/dataSet"+str(proId)+"/"+str(testId))
    #os.makedirs("C:/Users/zhang/Desktop/dataSet")
        
    startTime = time.clock()
        
    # make gripper opened
    inInts = [motorForce]
    inFloats = [motorVelocity1, motorVelocity2]
    res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_rightArm_joint1', vrep.sim_scripttype_childscript, 'pyGripper', inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    
    # go to initial position
    config = [0.87797689437866,-1.1669723987579,0.2386908531189,1.7083511352539,-0.10757255554199,1.0421299934387,0.36684370040894]
    moveToConfig(config, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold)
    startP = createDummy(baxterRightHand, baxterBaseHandle)
        
    #with open("C:/Users/zhang/Desktop/picturesRight"+str(proId)+"/"+str(testId)+"/dataRight.csv", "w", newline='') as file: 
    # with open("C:/Users/zhang/Desktop/dataSet/data.csv", "w", newline='') as file: 
    #     writer = csv.writer(file)
    #     writer.writerow(["obj", "picture", "bufferX", "bufferY", "graspX", "graspY", "graspOrientation", "isGrasp"])
    #     #writer.writerow(["obj", "picture", "graspX", "graspY", "graspOrientation", "isGrasp"])
        
    obj =  [0 for i in range(objectNum)]
    loopNum = 0
    scount = 0;
    count = 0;
    while time.clock()-startTime < 1200:
            loopNum += 1
            for i in range(objectNum):
                count += 1;
                print("", count)
                success = 1
                pid = loopNum * 10 + i
                res, obj[i] = vrep.simxGetObjectHandle(clientID, 'objr'+str(i), vrep.simx_opmode_blocking)
                error, graspPosition, graspOrientation, configBeforeRotate, aboveP, bufferX, bufferY = goAndGrasp(startP, rightCamera, pid, obj[i], graspHeight, gripperLength, offsetX, offsetY, regionY, regionZ, rightBox, baxterRightHand, baxterBaseHandle, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
                time.sleep(0.5)
                if error == 1:
                    continue
                if isGrasp(obj[i], tableHeight):
                    scount += 1;
                    success = 0
                    goAndPutDown(aboveP, regionY, regionZ, rightBox, baxterRightHand, baxterBaseHandle, maxVel, maxAccel, maxJerk, ikSteps, emptyBuff, pThreshold, oThreshold)
                #writer.writerow([obj[i], pid, bufferX, bufferY, graspPosition[0], graspPosition[1], graspOrientation, success])
                if count == 10:
                    print("successed: ", scount)
                    # Stop simulation:
                    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
                    # Now close the connection to V-REP:
                    vrep.simxFinish(clientID)
                    res=vrep.simxRemoveObject(clientID, aboveP, vrep.simx_opmode_blocking)
                #writer.writerow([obj[i], pid, graspPosition[0], graspPosition[1], graspOrientation, success])
                moveToConfig(configBeforeRotate, maxVel, maxAccel, maxJerk, fkSteps, emptyBuff, oThreshold)
                
                    
    
       
    
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
sys.exit("over")