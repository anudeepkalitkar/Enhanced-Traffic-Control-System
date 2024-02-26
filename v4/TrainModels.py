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
    "../../MSCourseProject Dataset/Boulder Traffic Cam Datasets/Annotations/"
)
WeatherImagesFolderPath = (
    "../../MSCourseProject Dataset/Boulder Traffic Cam Datasets/Images/"
)
processedAnnotationsFolder = "../../MSCourseProject Dataset/CarsDataset/Annotations/"
CarImagesFolderPath = "../../MSCourseProject Dataset/CarsDataset/Images/"


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
        monitor="val_mean_squared_error", patience=15, verbose=1, mode="min"
    )
    reduceLRPlateauCBK = ReduceLROnPlateau(
        monitor="val_mean_squared_error", factor=0.1, patience=5, verbose=1, mode="min"
    )
    modelCallbacks = [earlyStoppingCBK, reduceLRPlateauCBK]
    modelOptimizer = "adam"
    modelMetrics = ["mean_squared_error", "accuracy"]
    modelEpochs = 100
    modelBatchSize = 32
    modelInputShape = (540, 960)
    print("Creating Model")
    carDetection = CarDetection(
        modelCallbacks=modelCallbacks,
        modelOptimizer=modelOptimizer,
        modelMetrics=modelMetrics,
        modelEpochs=modelEpochs,
        modelBatchSize=modelBatchSize,
        modelInputShape=modelInputShape,
    )
    imageDataset, annotations = carDetection.LoadDataset(
        processedAnnotationsFolder, CarImagesFolderPath
    )
    processedDataset, processedAnnotations, imageWidth, imageHeight = (
        carDetection.PreProcessDataset(imageDataset, annotations)
    )
    
    carDetection.CreateNNModel()
    carDetection.DisplayNNSummary()
    carDetection.fit(processedDataset, processedAnnotations)
    
    # start = 0
    # for i in range(1,11):
    #     end = start + processedDataset.shape[0] // 10
    #     carDetection.fit(processedDataset[start:end], processedAnnotations[start:end])
    
    index = random.randint(0, len(processedDataset))
    
    for index in range(len(processedDataset)):
        testingImage = processedDataset[index]

        preds = carDetection.predict(np.array([testingImage]))[0]
        for i in range(41):
            if( preds[i]>0.5):
                print(processedAnnotations[index][i],1)
            else:
                print(processedAnnotations[index][i],0)
                
        bBoxes = []
        for i in range(0, len(processedAnnotations[index])//3):
            if processedAnnotations[index][i]>0.5:
                bBox = [
                    abs(processedAnnotations[index][i + 41] * imageHeight),
                    abs(processedAnnotations[index][i + 82 ] * imageWidth),
                ]
                bBoxes.append(bBox)

        BBtestingImage = DrawPoints(testingImage, bBoxes)

        bBoxes = []
        for i in range(0, 41):
            if preds[i]>0.5:
                bBox = [
                    preds[i + 41] * imageHeight,
                    preds[i + 82 ] * imageWidth,
                ]
                print(bBox)
                bBoxes.append(bBox)

        BBtestingImage = DrawPoints(BBtestingImage, bBoxes, (0,0,255))
        ShowImage("pred", BBtestingImage,)

    return carDetection

TrainCarDetection()