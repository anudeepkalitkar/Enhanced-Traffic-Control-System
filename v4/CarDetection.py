import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.losses import categorical_crossentropy, huber


class CarDetection(BaseEstimator):
    def __init__(
        self,
        BW_lowerWhite: list = [0, 0, 168],
        BW_upperWhite: list = [172, 111, 255],
        BW_thresh: int = 127,
        BW_maxval: int = 255,
        bilateralFilter_d: int = 11,
        bilateralFilter_sigmaColor: float = 17,
        bilateralFilter_sigmaSpace: float = 17,
        canny_lowerThreshold: float = 0,
        canny_upperThreshold: float = 0,
        NN_lossFunction=None,
        NN_optimizer: str = None,
        NN_metrics: list = None,
        NN_inputShape: tuple = None,
        maxNumVehicles: int = None,
        NN_epochs: int = None,
        NN_batchSize: int = None,
        NN_callbacks: list = None,
        cameraMargin: dict = None,
        NN_model: Sequential = None,
        # NN_lossFunction=None,
        # NN_optimizer: str = "adam",
        # NN_metrics: list = ["mean_squared_error", "accuracy"],
        # NN_inputShape: tuple = (540, 960),
        # maxNumVehicles: int = 123,
        # NN_epochs: int = 100,
        # NN_batchSize: int = 32,
        # NN_callbacks: list = [],
        # cameraMargin: dict = {},
        # NN_model: Sequential = None,
    ):
        self.BW_lowerWhite = BW_lowerWhite
        self.BW_upperWhite = BW_upperWhite
        self.BW_thresh = BW_thresh
        self.BW_maxval = BW_maxval
        self.bilateralFilter_d = bilateralFilter_d
        self.bilateralFilter_sigmaColor = bilateralFilter_sigmaColor
        self.bilateralFilter_sigmaSpace = bilateralFilter_sigmaSpace
        self.canny_lowerThreshold = canny_lowerThreshold
        self.canny_upperThreshold = canny_upperThreshold
        self.NN_lossFunction = NN_lossFunction
        self.NN_optimizer = NN_optimizer
        self.NN_metrics = NN_metrics
        self.NN_inputShape = NN_inputShape
        self.maxNumVehicles = maxNumVehicles
        self.NN_epochs = NN_epochs
        self.NN_batchSize = NN_batchSize
        self.NN_callbacks = NN_callbacks
        self.cameraMargin = cameraMargin
        self.NN_model = NN_model

    def LoadDataset(self, annotationsFolder: str, imagesFolderPath: str):
        if not (os.path.isdir(annotationsFolder) and os.path.isdir(imagesFolderPath)):
            print("Please ensure that annotationsFolder and imagesFolderPath are valid")
            return None
        annotationFileNames = os.listdir(annotationsFolder)
        imageDataset = []
        annotations = []
        for annotationFileName in tqdm(annotationFileNames):
            imageFileName = annotationFileName.split(".")[0] + ".png"
            if os.path.exists(imagesFolderPath + imageFileName):
                originalImage = cv2.imread(imagesFolderPath + imageFileName)
                originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

                with open(annotationsFolder + annotationFileName, "r") as jsonFile:
                    jsonData = json.load(jsonFile)
                bBoxes = []
                for carNumber, points in jsonData.items():
                    newPoints = []
                    for point in points:
                        newPoints.append([round(point[0]), round(point[1])])
                    bBoxes.append(newPoints)
                imageDataset.append(originalImage)
                annotations.append(bBoxes)

        return imageDataset, annotations

    def ResizeImage(self, capturedImage: np.ndarray, bBoxs: list, resizeScale=None):
        newDimentions = (self.NN_inputShape[1], self.NN_inputShape[0])
        if type(resizeScale) == type(int):
            newDimentions = (
                int(capturedImage.shape[1] * resizeScale / 100),
                int(capturedImage.shape[0] * resizeScale / 100),
            )
        resizedImage = cv2.resize(capturedImage, newDimentions)
        widthScale = newDimentions[0] / capturedImage.shape[1]
        heightScale = newDimentions[1] / capturedImage.shape[0]
        resizedBBoxs = []
        for bBox in bBoxs:
            x1 = int(bBox[0][0] * widthScale)
            y1 = int(bBox[0][1] * heightScale)
            x2 = int(bBox[1][0] * widthScale)
            y2 = int(bBox[1][1] * heightScale)
            resizedBBoxs.append([[x1, y1], [x2, y2]])

        return resizedImage, resizedBBoxs

    def ConvertDatasettoNumpy(self, imageList: list, annotationList: list):
        imageDataset = np.array(imageList)
        maxVehicles = 0
        for annotation in annotationList:
            if maxVehicles < len(annotation):
                maxVehicles = len(annotation)
        self.maxNumVehicles = maxVehicles 
        annotations = np.zeros((len(annotationList), maxVehicles*2, 2))
        height, width = imageDataset[0].shape[:2]
        
        for i in range(len(annotationList)):
            for j in range(maxVehicles):
                annotations[i,j] = 0
                
        for i, bBoxAnnotation in enumerate(annotationList):
            for j, bBox in enumerate(bBoxAnnotation):
                if j < maxVehicles:
                    x = abs(bBox[0][0] + bBox[1][0]) / (2 * height)
                    y = abs(bBox[0][1] + bBox[1][1]) / (2 * width)
                    annotations[i, maxVehicles+j] = [ x, y]
                    annotations[i, j] = 1

        annotations = annotations.reshape(len(annotationList), -1)
        return imageDataset, annotations

    def PreProcessDataset(self, imageDataset: list, annotations: list):
        resizedImages = []
        resizedAnnotations = []
        for i in range(len(imageDataset)):
            resizedImage, resizedBBoxs = self.ResizeImage(
                imageDataset[i], annotations[i]
            )
            resizedImages.append(resizedImage)
            resizedAnnotations.append(resizedBBoxs)

        height, width = resizedImages[0].shape[:2]
        processedDataset, processedAnnotations = (
            self.ConvertDatasettoNumpy(resizedImages, resizedAnnotations)
        )
        # nonMajorColorImage = self.RemoveMajorColor(capturedImage)
        
        return processedDataset, processedAnnotations, width, height

    def CannyEdgeConverstion(self, capturedImage: np.ndarray):
        bilateralFilter = cv2.bilateralFilter(
            capturedImage,
            self.bilateralFilter_d,
            self.bilateralFilter_sigmaColor,
            self.bilateralFilter_sigmaSpace,
        )
        # imageMedian = np.median(capturedImage)
        # lowerThreshold = max(0, (0.7 * imageMedian))
        # upperThreshold = min(255, (0.7 * imageMedian))
        cannyEdgeImage = cv2.Canny(
            bilateralFilter, self.canny_lowerThreshold, self.canny_upperThreshold
        )
        return cannyEdgeImage

    def BlackWhiteConverstion(self, originalImage: np.ndarray):
        hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
        # lowerWhite = np.array([0, 0, 168])
        # upperWhite = np.array([172, 111, 255])
        mask = cv2.inRange(
            hsv, np.array(self.BW_lowerWhite), np.array(self.BW_upperWhite)
        )
        masked = cv2.bitwise_and(originalImage, originalImage, mask=mask)
        (thresh, blackWhiteImage) = cv2.threshold(
            masked, self.BW_thresh, self.BW_maxval, cv2.THRESH_BINARY
        )
        return blackWhiteImage

    def ExtractImageFeatures(self, capturedImage: np.ndarray):
        # nonMajorColorImage = cv2.cvtColor(capturedImage,cv2.COLOR_BGR2GRAY)
        blackWhiteImage = self.BlackWhiteConverstion(capturedImage)
        cannyEdgeImage = self.CannyEdgeConverstion(blackWhiteImage)

        # cv2.imshow("nonMajorColorImage", nonMajorColorImage)
        # cv2.imshow("blackWhiteImage", blackWhiteImage)
        # cv2.imshow("cannyEdgeImage", cannyEdgeImage)
        # cv2.waitKey()
        return cannyEdgeImage

    def CreateNNModel(self):
        inputLayer = Input(shape=(self.NN_inputShape[0], self.NN_inputShape[1], 1))
        layer1 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="valid",
        )(inputLayer)
        layer1 = MaxPool2D((2, 2))(layer1)

        layer2 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="valid",
        )(layer1)
        layer2 = MaxPool2D((2, 2))(layer2)

        layer3 = Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="valid",
        )(layer2)
        layer3 = MaxPool2D((2, 2))(layer3)

        flattenLayer = Flatten()(layer3)

        classificationOutput = Dense(
            self.maxNumVehicles, activation="softmax", name="classification_output"
        )(flattenLayer)
        bboxOutput = Dense(4, activation="linear", name="bbox_output")(flattenLayer)

        NN_model = Model(inputs=inputLayer, outputs=[classificationOutput, bboxOutput])

        NN_model.compile(
            loss=self.NN_lossFunction,
            optimizer=self.NN_optimizer,
            metrics=self.NN_metrics,
        )
        return NN_model

    def CustomLossFunction(self, y_true, y_pred):
        y_true_cls, y_true_bbox = (
            y_true[..., : self.maxNumVehicles],
            y_true[..., self.maxNumVehicles :],
        )
        y_pred_cls, y_pred_bbox = (
            y_pred[..., : self.maxNumVehicles],
            y_pred[..., self.maxNumVehicles :],
        )
        classificationLoss = categorical_crossentropy(y_true_cls, y_pred_cls)
        bboxLoss = huber(y_true_bbox, y_pred_bbox)
        totalLoss = classificationLoss + bboxLoss
        return totalLoss

    def TrainNNModel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.NN_model.fit(
            X_train,
            y_train,
            self.NN_batchSize,
            self.NN_epochs,
            validation_split=0.2,
            callbacks=self.NN_callbacks,
        )

    def fit(self, X, y):
        print("PreProcessing X")
        X_processed = [self.ExtractImageFeatures(image) for image in tqdm(X)]
        X_processed = np.array(X_processed)
        y_processed = np.array(y)

        if len(X_processed.shape) == 3:
            X_processed = X_processed.reshape(
                X_processed.shape[0], X_processed.shape[1], X_processed.shape[2], 1
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        self.TrainNNModel(X_train, y_train)
        return self

    def predict(self, X, y=None):
        X_processed = [self.PreProcessImage(image) for image in tqdm(X)]
        X_processed = np.array(X_processed)

        if len(X_processed.shape) == 3:
            X_processed = X_processed.reshape(
                X_processed.shape[0], X_processed.shape[1], X_processed.shape[2], 1
            )

        return self.NN_model.predict(X_processed)

    def DisplayNNSummary(self):
        self.NN_model.summary()
