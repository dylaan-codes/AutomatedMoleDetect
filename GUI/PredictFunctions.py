import cv2
from ultralytics import YOLO
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os


def openFile():
    try:
        filepath = filedialog.askopenfilename(initialdir="./GUI/Model3/images",
                                              title="Open image",
                                              filetypes=(
                                              ("image files", "*.JPG *.PNG *.JPEG"), ("JPG", "*.JPG"), ("PNG", "*.PNG"),
                                              ("JPEG", "*.JPEG")))  # Filter so only image files can be selected
        return filepath
    except:
        # Display alert that unreadable image format
        messagebox.showerror("showerror", "Something went wrong reading the image")
        print("Something went wrong reading the image")


class YOLOsegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def Predictions(self, image):
        global SegmentedImage
        global MaskImage
        global AnnotatedImage
        global BboxImage
        global Coordinates

        try:
            # Get shape of image, you need this to resize the image to the original dimensions
            H, W, _ = image.shape
            results = self.model.predict(source=image.copy(), save=False,
                                         save_txt=False, conf=0.6)  # Results is an object with the prediction data

            annotated_frame = results[0].plot()
            AnnotatedImage = annotated_frame  # [1] Get the first display image

            if results[0].masks is not None:  # if the model has detected a lesion(s)
                MaskImage = np.zeros((H, W), dtype=np.uint8)
                for i in range(len(results[0].masks)):  # For the number of lesions in the image we combine to create a single mask.
                    RawMask = results[0].masks[i].cpu().data.numpy().transpose(1, 2, 0)  # A greyscale image
                    tempMask = cv2.resize(RawMask, (W, H))
                    tempMask = (tempMask * 255).astype(np.uint8)
                    MaskImage = cv2.bitwise_or(MaskImage, tempMask)  # [2] Mask Image, Add new mask to Combined Masks
                SegmentedImage = cv2.bitwise_and(image, image, mask=MaskImage)  # [3] SegmentedImage, Using the mask anything

            for result in results:
                bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # Array of bounding box co-ordinates
                conf_scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)  # Confidence score
                Coordinates = bboxes

                Bbox_Temp = image.copy()
                for bbox, score in zip(bboxes, conf_scores):
                    #print("bbox:", bbox, "score:", score)
                    (x, y, x2, y2) = bbox
                    Bbox_Temp = cv2.rectangle(Bbox_Temp, (x, y), (x2, y2), (255, 0, 0), 2)

                BboxImage = Bbox_Temp

            DetectBool = True  # Check if any mole has been detected
            if len(Coordinates)==0:
                DetectBool = False
            return AnnotatedImage, MaskImage, SegmentedImage, BboxImage, Coordinates, DetectBool
        except:
            pass

def getabsPath(relP):
    AbsolutePath = os.path.abspath(relP)  # Absolute path of model weight
    AbsolutePath = AbsolutePath.replace("\\", r"/")
    return AbsolutePath


#DetectIt = YOLOsegmentation("E:\\MOLEDETECTION\\PROGRAM\\GUI\\Model3\\runs\\segment\\train\\weights\\best.pt")
#OGpath = openFile()
#OriginalImage = cv2.imread(OGpath)
#OGcopy = OriginalImage

#Coordinates = DetectIt.Predictions(OriginalImage)
