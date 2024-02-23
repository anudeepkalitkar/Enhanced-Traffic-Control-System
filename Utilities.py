import os
import cv2
import numpy as np
from keras.models import Sequential
import copy

modelDir = "./SavedModels/"
if not os.path.exists(modelDir):
    os.mkdir(modelDir)
    
def SaveModels(model: Sequential, modelName: str, modelDir: str = modelDir):
    try:
        model.save(modelDir + modelName + ".h5")
        return True
    except Exception as e:
        print("Could Not save the model with Exception:", e)
        return False
    
def ShowImage( imageName: str, image: np.ndarray):
    cv2.imshow(imageName, image)
    cv2.waitKey(0)
    return None

def DisplayPrediction(image: np.ndarray, weather: str = None):
    if not weather:
        weather = "Cannot Predict"
    image_width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10
    font_color = (0, 255, 0)  # Green color
    line_type = 2
    text_size = cv2.getTextSize(weather, font, font_scale, line_type)[0]
    text_x = (image_width - text_size[0]) // 2  # Centered horizontally
    text_y = 30
    image = cv2.putText(
        image, weather, (text_x, text_y), font, font_scale, font_color, line_type
    )
    return image



def DrawBoundaryBoxs(
    image: np.ndarray,
    boundryBoxes: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    boundaryBoxImage = copy.deepcopy(image)
    for [[x1, y1], [x2, y2]] in boundryBoxes:
        boundaryBoxImage = cv2.rectangle(
            boundaryBoxImage, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
        )
    return boundaryBoxImage