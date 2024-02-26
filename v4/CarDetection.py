import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.losses import binary_crossentropy, huber


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
        modelOptimizer: str = None,
        modelMetrics: list = None,
        modelInputShape: tuple = None,
        maxNumVehicles: int = None,
        modelEpochs: int = None,
        modelBatchSize: int = None,
        modelCallbacks: list = None,
        cameraMargin: dict = None,
        modelPath: str = None,
        # NN_lossFunction=None,
        # modelOptimizer: str = "adam",
        # modelMetrics: list = ["mean_squared_error", "accuracy"],
        # modelInputShape: tuple = (540, 960),
        # maxNumVehicles: int = 123,
        # modelEpochs: int = 100,
        # modelBatchSize: int = 32,
        # modelCallbacks: list = [],
        # cameraMargin: dict = {},
        # model: Sequential = None,
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
        self.modelOptimizer = modelOptimizer
        self.modelMetrics = modelMetrics
        self.modelInputShape = modelInputShape
        self.maxNumVehicles = maxNumVehicles
        self.modelEpochs = modelEpochs
        self.modelBatchSize = modelBatchSize
        self.modelCallbacks = modelCallbacks
        self.cameraMargin = cameraMargin
        if modelPath != None:
            self.model = load_model(modelPath, custom_objects=self.CustomLossFunction)

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
        newDimentions = (self.modelInputShape[1], self.modelInputShape[0])
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

        annotations = np.zeros((len(annotationList), 3, maxVehicles))
        height, width = imageDataset[0].shape[:2]

        for i, bBoxAnnotation in enumerate(annotationList):
            for j, bBox in enumerate(bBoxAnnotation):
                if j < maxVehicles:
                    # x1 = bBox[0][0] / height
                    # y1 = bBox[0][1] / width
                    # x2 = bBox[1][0] / height
                    # y2 = bBox[1][1] / width
                    x = abs(bBox[0][0] + bBox[1][0]) / (2 * height)
                    y = abs(bBox[0][1] + bBox[1][1]) / (2 * width)
                    annotations[i, 0, j] = 1
                    annotations[i, 1, j ] = x
                    annotations[i, 2 , j] = y
                    # annotations[i, j + maxVehicles + 2 + j] = x2
                    # annotations[i, j + maxVehicles + 3 + j] = y2

        annotations = annotations.reshape(len(annotationList), -1)
        return imageDataset, annotations


    def is_not_white(self, color, threshold=220):
        return np.all(color < threshold)
    
    
    def FindMajorityColor(self, processedDataset: np.ndarray, n_clusters: int = 3):
        pixels = processedDataset.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        sortedIndices = np.argsort(-counts)
        clusterCenters = kmeans.cluster_centers_[sortedIndices]
        nonWhiteCenters = [
            center for center in clusterCenters if self.is_not_white(center)
        ]
        majorColor = nonWhiteCenters[0] if nonWhiteCenters else clusterCenters[0]
        return majorColor
    
    def RemoveMajorColor(self, processedDataset: np.ndarray, threshold: int = 40):
        nonMajorColorImages = []
        majorColor = self.FindMajorityColor(processedDataset)
        lowerThreshold = np.maximum(majorColor - threshold, 0)
        upperThreshold = np.minimum(majorColor + threshold, 255)
        blankWhiteImage = np.full(processedDataset[0].shape, 255, dtype=np.uint8)
        for image in tqdm(processedDataset):
            majorColorMask = cv2.inRange(image, lowerThreshold, upperThreshold)
            nonMajorColorImage = cv2.bitwise_and(image, image, mask=~majorColorMask)
            majorColortoWhiteImage = cv2.bitwise_and(
                blankWhiteImage, blankWhiteImage, mask=majorColorMask
            )
            nonMajorColorImage = cv2.add(
                nonMajorColorImage, majorColortoWhiteImage
            )  # Combine the two
            nonMajorColorImages.append(nonMajorColorImage)
        
        return np.array(nonMajorColorImages)
            
        
    def PreProcessDataset(self, imageDataset: list, annotations: list):
        resizedImages = []
        resizedAnnotations = []
        for i in tqdm(range(len(imageDataset))):
            resizedImage, resizedBBoxs = self.ResizeImage(
                imageDataset[i], annotations[i]
            )
            resizedImages.append(resizedImage)
            resizedAnnotations.append(resizedBBoxs)

        height, width = resizedImages[0].shape[:2]
        processedDataset, processedAnnotations = self.ConvertDatasettoNumpy(
            resizedImages, resizedAnnotations
        )
        print("Removeing Major Color")
        # nonMajorColorImages = self.RemoveMajorColor(processedDataset)
        # print(nonMajorColorImages.shape)
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
        # cannyEdgeImage = cv2.Canny(
        #     bilateralFilter, lowerThreshold, upperThreshold
        # )
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

        # cv2.imshow("nonMajorColorImage", capturedImage)
        # cv2.imshow("blackWhiteImage", blackWhiteImage)
        # cv2.imshow("cannyEdgeImage", cannyEdgeImage)
        # cv2.waitKey()
        return cannyEdgeImage

    def CreateNNModel(self):
        inputLayer = Input(shape=(self.modelInputShape[0], self.modelInputShape[1], 1))
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
            self.maxNumVehicles, activation="sigmoid", name="classificationOutput"
        )(flattenLayer)
    
        bboxOutput = Dense(
          (  self.maxNumVehicles * 2), activation="linear", name="bboxOutput"
        )(flattenLayer)
        model = Model(inputs=inputLayer, outputs=[tf.concat([classificationOutput,bboxOutput],axis=1) ])

        model.compile(
            loss=self.CustomLossFunction,
            optimizer=self.modelOptimizer,
            metrics=self.modelMetrics,
        )
        self.model = model
        return True

    def CustomLossFunction(self, y_true, y_pred):
        y_true_cls, y_true_bbox = tf.split(y_true, [self.maxNumVehicles, self.maxNumVehicles * 2], axis=-1)
        y_pred_cls, y_pred_bbox = tf.split(y_pred, [self.maxNumVehicles, self.maxNumVehicles * 2], axis=-1)
 
        classificationLoss = binary_crossentropy(y_true_cls, y_pred_cls)
        bboxLoss = huber(y_true_bbox, y_pred_bbox)
        totalLoss = classificationLoss + bboxLoss
        return totalLoss

    def TrainNNModel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.model.fit(
            X_train,
            y_train,
            self.modelBatchSize,
            self.modelEpochs,
            validation_split=0.2,
            callbacks=self.modelCallbacks,
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

        # self.TrainNNModel(X_processed, y_processed)
        self.TrainNNModel(X_train, y_train)
        return self

    def predict(self, X, y=None):
        X_processed = [self.ExtractImageFeatures(image) for image in tqdm(X)]
        X_processed = np.array(X_processed)

        if len(X_processed.shape) == 3:
            X_processed = X_processed.reshape(
                X_processed.shape[0], X_processed.shape[1], X_processed.shape[2], 1
            )

        return self.model.predict(X_processed)
    
    def evaluate(self, X, y):
        X_processed = [self.ExtractImageFeatures(image) for image in tqdm(X)]
        X_processed = np.array(X_processed)
        y_processed = np.array(y)

        if len(X_processed.shape) == 3:
            X_processed = X_processed.reshape(
                X_processed.shape[0], X_processed.shape[1], X_processed.shape[2], 1
            )
        history = self.model.evaluate(X_processed, y_processed)
        return history
        
        
    def DisplayNNSummary(self):
        self.model.summary()
