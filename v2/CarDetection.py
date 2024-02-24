import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.losses import Huber


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
        NN_lossFunction=Huber(delta=1.0),
        NN_optimizer: str = "adam",
        NN_metrics: list = ["mean_squared_error", "accuracy"],
        NN_inputShape: tuple = (960, 540),
        NN_numberofOutputNeurons: int = 164,
        NN_epochs: int = 100,
        NN_batchSize: int = 32,
        NN_callbacks: list = [],
        cameraMargin: dict = {},
        NN_model: Sequential = None,
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
        self.NN_numberofOutputNeurons = NN_numberofOutputNeurons
        self.NN_epochs = NN_epochs
        self.NN_batchSize = NN_batchSize
        self.NN_callbacks = NN_callbacks
        self.cameraMargin = cameraMargin
        if NN_model is None:
            self.NN_model = self.CreateNNModel()
        else:
            self.NN_model = NN_model

    def FindMaxDimensions(self, imageList: list):
        maxWidth = 0
        maxHeight = 0
        for img in imageList:
            h, w = img.shape[:2]
            if h > maxHeight:
                maxHeight = h
            if w > maxWidth:
                maxWidth = w
        return maxWidth, maxHeight

    def ConvertDatasettoNumpy(self, imageList: list, annotationList: list):

        imageDataset = np.array(imageList)
        maxLines = 0
        for annotation in annotationList:
            if maxLines < len(annotation):
                maxLines = len(annotation)
        annotations = np.zeros((len(annotationList), maxLines, 4))
        height, width = imageDataset[0].shape[:2]

        for i, bBoxAnnotation in enumerate(annotationList):
            for j, bBox in enumerate(bBoxAnnotation):
                if j < maxLines:
                    x1 = bBox[0][0] / height
                    y1 = bBox[0][1] / width
                    x2 = bBox[1][0] / height
                    y2 = bBox[1][1] / width
                    annotations[i, j] = [x1, y1, x2, y2]

        annotations = annotations.reshape(len(annotationList), -1)
        return imageDataset, annotations, width, height

    def ResizeImage(
        self, capturedImage: np.ndarray, bBoxs: list, resizeScale = None
    ):
        newDimentions = (self.NN_inputShape[1], self.NN_inputShape[0])
        if type(resizeScale) == type(int):
            newDimentions = (
                int(capturedImage.shape[1] * resizeScale / 100),
                int(capturedImage.shape[0] * resizeScale / 100),
            )
        resizedImage = cv2.resize(capturedImage, newDimentions)
        widthScale = newDimentions[0] / capturedImage.shape[1]
        heightScale = newDimentions[1] / capturedImage.shape[0]
        newbBoxs = []
        for bBox in bBoxs:
            x1 = int(bBox[0][0] * widthScale)
            y1 = int(bBox[0][1] * heightScale)
            x2 = int(bBox[1][0] * widthScale)
            y2 = int(bBox[1][1] * heightScale)
            newbBoxs.append([[x1, y1], [x2, y2]])

        return resizedImage, newbBoxs

    def LoadDataset(self, annotationsFolder: str, imagesFolderPath: str):
        if not (os.path.isdir(annotationsFolder) and os.path.isdir(imagesFolderPath)):
            print("Please ensure that annotationsFolder and imagesFolderPath are valid")
            return None

        annotationFileNames = os.listdir(annotationsFolder)
        imageDataset = []
        annotations = []
        for annotationFileName in tqdm(annotationFileNames):
            imageFileName = annotationFileName.split(".")[0] + ".png"
            if(os.path.exists(imagesFolderPath + imageFileName)):
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
                resizedImage, newbBoxs = self.ResizeImage(originalImage, bBoxes)
                imageDataset.append(resizedImage)
                annotations.append(newbBoxs)

        imageDataset, annotations, width, height = self.ConvertDatasettoNumpy(
            imageDataset, annotations
        )

        return imageDataset, annotations, width, height

    def fit(self, X, y):
        print("PreProcessing X")
        X_processed = [self.PreProcessImage(image) for image in tqdm(X)]
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

    def PreProcessImage(self, capturedImage: np.ndarray):
        
        # nonMajorColorImage = self.RemoveMajorColor(capturedImage)
        nonMajorColorImage = cv2.cvtColor(capturedImage,cv2.COLOR_BGR2GRAY) 
        # cv2.imshow("nonMajorColorImage", nonMajorColorImage)
        # nonMajorColorImage = cv2.cvtColor(nonMajorColorImage,cv2.COLOR_GRAY2BGR)
        # blackWhiteImage = self.BlackWhiteConverstion(nonMajorColorImage)
        # cv2.imshow("blackWhiteImage", blackWhiteImage)
        
        # cannyEdgeImage = self.CannyEdgeConverstion(blackWhiteImage)
        # cv2.imshow("cannyEdgeImage", cannyEdgeImage)
        # cv2.waitKey()
        return nonMajorColorImage

    def CannyEdgeConverstion(self, capturedImage: np.ndarray):
        # grayScale = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2GRAY)
        bilateralFilter = cv2.bilateralFilter(
            # grayScale,
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

    def is_not_white(self, color, threshold=220):
        return np.all(color < threshold)

    def FindMajorityColor(self, image: np.ndarray, n_clusters: int = 3):
        pixels = image.reshape((-1, 3))
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

    def CreateMajorColorMask(self, image: np.ndarray, majorColor, threshold: int = 40):
        lowerThreshold = np.maximum(majorColor - threshold, 0)
        upperThreshold = np.minimum(majorColor + threshold, 255)
        majorColorMask = cv2.inRange(image, lowerThreshold, upperThreshold)

        return majorColorMask

    def RemoveMajorColor(self, image: np.ndarray):
        majorColor = self.FindMajorityColor(image)
        majorColorMask = self.CreateMajorColorMask(image, majorColor)
        blankWhiteImage = np.full(image.shape, 255, dtype=np.uint8)

        nonMajorColorImage = cv2.bitwise_and(image, image, mask=~majorColorMask)
        majorColortoWhiteImage = cv2.bitwise_and(
            blankWhiteImage, blankWhiteImage, mask=majorColorMask
        )
        nonMajorColorImage = cv2.add(
            nonMajorColorImage, majorColortoWhiteImage
        )  # Combine the two

        return nonMajorColorImage

    def CreateNNModel(self):
        NN_model = Sequential()
        NN_model.add(
            Conv2D(
                16,
                (3, 3),
                activation="relu",
                padding="valid",
                input_shape=(self.NN_inputShape[0], self.NN_inputShape[1], 1),
            )
        )
        NN_model.add(MaxPool2D((2, 2)))

        NN_model.add(Conv2D(32, (3, 3), activation="relu", padding="valid"))
        NN_model.add(MaxPool2D((2, 2)))

        NN_model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
        NN_model.add(MaxPool2D((2, 2)))

        NN_model.add(Flatten())
        NN_model.add(Dense(128, activation="relu"))
        # NN_model.add(Dropout(0.5))
        NN_model.add(Dense(self.NN_numberofOutputNeurons))

        NN_model.compile(
            loss=self.NN_lossFunction,
            optimizer=self.NN_optimizer,
            metrics=self.NN_metrics,
        )
        return NN_model

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

    def DisplayNNSummary(self):
        self.NN_model.summary()
