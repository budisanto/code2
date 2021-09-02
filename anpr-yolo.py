# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
#import pytesseract
import easyocr
import skewing2

reader = easyocr.Reader(['en'], gpu = False)
# Initialize the parameters
confThreshold = 0.3  # Confidence threshold
nmsThreshold = 0.2  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "darknet-yolov3.cfg"
modelWeights = "model.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Process inputs
winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    
    cropped = frame
    #cv.imshow('Out',cropped)
    #cv.waitKey(0)

    # Stop the program if reached end of video
    if not hasFrame:
        #print("Done processing !!!")
        #print("Output file is stored as ", outputFile)
        #cv.waitKey(0)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False) #False original

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    #postprocess(frame, outs)
    # Remove the bounding boxes with low confidence using non-maxima suppression
    #=============================================================
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                #print(detection[4], " - ", scores[classId]," - th : ", confThreshold)
                confThreshold = confThreshold
                #print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
    #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    #=============================================================
    #def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    #cv.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)
    
    #cropped = cropped[left:right,top:bottom]
   
    
    label = '%.2f' % confidences[i]

    # Get the label for the class name and its confidence
    if classes:
        assert(classIds[i] < len(classes))
        label = '%s: %s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    #cv.putText(frame, label, (left, top),cv.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    #=============================================================
    
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #print(box)
    # Write the frame with the detection boxes
    if (args.image):
        #cv.imwrite(outputFile, frame.astype(np.uint8))
        #image = frame.astype(np.uint8)
        y1 = box[1] - 10
        y2 = y1 + box[3] + 20
        x1 = box[0] - 10
        x2 = x1 + box[2] + 20
        cropped = cropped[y1:y2,x1:x2]
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        cropped = cv.resize(gray,(195,50))
        #cropped = cv.resize(gray,(600,150))
        #cropped = cv.GaussianBlur(cropped,(11,11),0)
        #cropped = cv.medianBlur(cropped,11)
        #cropped = cv.equalizeHist(cropped)
        
        # apply threshold
        #im = cv.normalize(cropped, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        #cv.imshow('Crop',cropped)
        #cv.imshow('BW',im)
        #cv.imshow('Out1',image)
        #cv.waitKey(0)
        cropped = skewing2.hori(cropped)
        print(reader.readtext(cropped, detail = 0))
        #cv.imwrite('out.jpg', cropped.astype(np.uint8))
        #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        #text = pytesseract.image_to_string(cropped, config='--psm 3')
        #print("Detected Number is:",text)        
        #print("Bounding box:",bounding) 
    else:
        vid_writer.write(frame.astype(np.uint8))