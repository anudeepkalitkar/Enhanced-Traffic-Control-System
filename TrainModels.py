import warnings
import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from Utilities import *

warnings.filterwarnings("ignore")
physicalDevices = tf.config.list_physical_devices("GPU")
print(physicalDevices)
if len(physicalDevices) > 0:
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)

earlyStoppingCBK = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")
reduceLRPlateauCBK = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=7, verbose=1, mode="min"
)
modelCallbacks = [earlyStoppingCBK, reduceLRPlateauCBK]


from WeatherPrediction import WeatherPrediction

WeatherAnnotationsFolder = "./Datasets/Boulder Traffic Cam Datasets/Annotations/"
WeatherImagesFolderPath = "./Datasets/Boulder Traffic Cam Datasets/Images/"

def TrainWeather():
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
        weatherModel, X_train, y_train, 100, 32, modelCallbacks
    )
    print("Validating Weather Model")
    weatherClass.ValidateModel(weatherModel, X_test, y_test)
    modelSaved = SaveModels(weatherModel, "WeatherPredictModel")
    if(not modelSaved):
        return False

