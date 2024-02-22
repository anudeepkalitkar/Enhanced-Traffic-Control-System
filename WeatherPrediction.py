import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


class WeatherPrediction:
    def __init__(self):
        
        pass
    def BlackWhiteConverstion(self, originalImage: np.ndarray):
        hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
        lowerWhite = np.array([0, 0, 168])
        upperWhite = np.array([172, 111, 255])
        mask = cv2.inRange(hsv, lowerWhite, upperWhite)
        masked = cv2.bitwise_and(originalImage, originalImage, mask=mask)
        (thresh, blackWhiteImage) = cv2.threshold(masked, 127, 255, cv2.THRESH_BINARY)
        return blackWhiteImage

    def ImagePreprocessing(
        self,
        originalImage: np.ndarray,
        resizeScale: int = 25,
        cameraMargin: dict = {"width": 0, "height": 150},
    ):
        newDimentions = (
            int(originalImage.shape[1] * resizeScale / 100),
            int(originalImage.shape[0] * resizeScale / 100),
        )
        originalImage = originalImage[cameraMargin["height"] : -cameraMargin["height"]]
        resizedImage = cv2.resize(originalImage, newDimentions)
        bwImage = self.BlackWhiteConverstion(resizedImage)
        return bwImage
    
    def LoadDataSet(self, datasetFolder: str):
        imageNames = os.listdir(datasetFolder)
        x = []
        y = []
        for imageName in tqdm(imageNames):
            originalImage = cv2.imread(datasetFolder + imageName)
            bwImage = self.ImagePreprocessing(originalImage)
            x.append(bwImage)
            if "Night" in imageName:
                y.append(1)
            else:
                y.append(0)
        x = np.array(x)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    
    def CreateWeatherPredictionModel(
        self, lossFunction: str, optimizer: str, metrics: list, inputShape: tuple
    ):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="valid",
                input_shape=(inputShape[0], inputShape[1], inputShape[2]),
            )
        )
        model.add(MaxPool2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
        model.add(MaxPool2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu", padding="valid"))
        model.add(MaxPool2D((2, 2)))

        model.add(Flatten())
        # model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss=lossFunction, optimizer=optimizer, metrics=metrics)
        return model
    
    def Train(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batchSize: int,
        callbacks: list,
    ):
        model.fit(
            X_train,
            y_train,
            batchSize,
            epochs,
            validation_split=0.2,
            callbacks=callbacks,
        )
        return model
    
    def ValidateModel(self, model: Sequential, X_test: np.ndarray, y_test: np.ndarray):
        model.evaluate(X_test, y_test)
        return None
    
    def Predict(self, model: Sequential, images: np.ndarray):
        predictions = []
        if(len(images.shape) >3):
            for image in images:
                bwImage = self.ImagePreprocessing(image)
                prediction = model.predict(bwImage.reshape((1, bwImage.shape[0],  bwImage.shape[1], bwImage.shape[2])))
                predictions.append(prediction[-1][-1])
                
        else:
            bwImage = self.ImagePreprocessing(images)
            prediction = model.predict(bwImage.reshape((1, bwImage.shape[0],  bwImage.shape[1], bwImage.shape[2])))
            predictions.append(prediction[-1][-1])
        return predictions