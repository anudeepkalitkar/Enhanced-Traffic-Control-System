import warnings
import random

import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from Utilities import *
from keras.models import load_model

warnings.filterwarnings("ignore")
physicalDevices = tf.config.list_physical_devices("GPU")
print(physicalDevices)
if len(physicalDevices) > 0:
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)


from WeatherPrediction import WeatherPrediction
from CarDetection import CarDetection

WeatherAnnotationsFolder = (
    "../MSCourseProject Dataset/Boulder Traffic Cam Datasets/Annotations/"
)
WeatherImagesFolderPath = (
    "../MSCourseProject Dataset/Boulder Traffic Cam Datasets/Images/"
)
CarAnnotationsFolder = "../MSCourseProject Dataset/CarsDataset/Annotations/"
CarImagesFolderPath = "../MSCourseProject Dataset/CarsDataset/Images/"


def TrainWeatherPrediction():

    earlyStoppingCBK = EarlyStopping(
        monitor="val_accuracy", patience=15, verbose=1, mode="auto"
    )
    reduceLRPlateauCBK = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.1, patience=5, verbose=1, mode="auto"
    )
    WeatherModelCallbacks = [earlyStoppingCBK, reduceLRPlateauCBK]

    weatherClass = WeatherPrediction()
    weatherLossFunction = "mean_squared_error"
    weatherOptimizer = "adam"
    WeatherMetrics = ["accuracy", "mean_squared_error"]
    print("Loading DataSet")
    X_train, X_test, y_train, y_test = weatherClass.LoadDataSet(WeatherImagesFolderPath)
    print("Creating Weather Model")
    weatherModel = weatherClass.CreateWeatherPredictionModel(
        weatherLossFunction, weatherOptimizer, WeatherMetrics, X_train[0].shape
    )
    print("Training Weather Model")
    weatherModel = weatherClass.Train(
        weatherModel, X_train, y_train, 100, 32, WeatherModelCallbacks
    )

    modelSaved = SaveModels(weatherModel, "WeatherPredictModel")
    if modelSaved:
        print("Model Trained and Saved !!!!")


def ValidateWeatherPrediction():
    weatherClass = WeatherPrediction()
    X_train, X_test, y_train, y_test = weatherClass.LoadDataSet(WeatherImagesFolderPath)
    weatherModel = load_model(modelDir + "WeatherPredictModel.h5")
    truePrediction = 0
    for i in range(len(X_test)):
        prediction = weatherModel.predict(
            X_test[i].reshape(
                (1, X_test[i].shape[0], X_test[i].shape[1], X_test[i].shape[2])
            ),
            verbose=0,
        )
        if int(prediction[-1][-1]) == y_test[i]:
            truePrediction += 1
    print(truePrediction / len(X_test) * 100)


def TrainCarDetection():
    earlyStoppingCBK = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, mode="min"
    )
    reduceLRPlateauCBK = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min"
    )
    modelCallbacks = [earlyStoppingCBK, reduceLRPlateauCBK]

    print("Creating Model")
    carDetection = CarDetection(NN_callbacks=modelCallbacks)
    print("Loading DataSet")
    carImageDataset, carAnnotations, imageWidth, imageHeight = carDetection.LoadDataset(
        CarAnnotationsFolder, CarImagesFolderPath
    )
    start = 0
    print("Training Model")
    for i in range(1, 11):
        end = start + carImageDataset.shape[0] // 10
        carDetection.fit(carImageDataset[start:end], carAnnotations[start:end])
        start = end

    modelSaved = SaveModels(carDetection.NN_model, "CarDetectionModel")
    if modelSaved:
        print("Model Trained and Saved !!!!")


def ValidateCarDetection():
    earlyStoppingCBK = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, mode="min"
    )
    reduceLRPlateauCBK = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min"
    )
    modelCallbacks = [earlyStoppingCBK, reduceLRPlateauCBK]
    NN_model = load_model("SavedModels/CarDetectionModel.h5")
    carDetection = CarDetection(NN_model=NN_model, NN_callbacks=modelCallbacks)
    carImageDataset, carAnnotations, imageWidth, imageHeight = carDetection.LoadDataset(
        CarAnnotationsFolder, CarImagesFolderPath
    )

    index = random.randint(0, len(carImageDataset))
    testingImage = carImageDataset[index]

    preds = carDetection.predict(np.array([testingImage]))[0]
    print(len(carAnnotations[index]), len(preds))
    print(carAnnotations[index])
    print(preds)

    bBoxes = []
    for i in range(0, len(carAnnotations[index]), 4):
        bBox = [
            [
                carAnnotations[index][i] * imageHeight,
                carAnnotations[index][i + 1] * imageWidth,
            ],
            [
                carAnnotations[index][i + 2] * imageHeight,
                carAnnotations[index][i + 3] * imageWidth,
            ],
        ]
        bBoxes.append(bBox)
    BBtestingImage = DrawBoundaryBoxs(testingImage, bBoxes)
    ShowImage("test", BBtestingImage)
    bBoxes = []
    for i in range(0, len(preds), 4):
        bBox = [
            [preds[i] * imageWidth, preds[i + 1] * imageHeight],
            [preds[i + 2] * imageWidth, preds[i + 3] * imageHeight],
        ]
        bBoxes.append(bBox)
    BBtestingImage = DrawBoundaryBoxs(testingImage, bBoxes)
    ShowImage("test", BBtestingImage)


TrainWeatherPrediction()
ValidateWeatherPrediction()
TrainCarDetection()
ValidateCarDetection()