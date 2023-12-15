import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = load_model("/mnt/vol_1/waste_clf_tracking/mlruns/799454089708736406/3a8dfd55898746719fdade117bc58a5d/artifacts/MultiBaseTracking-1Run-2waste_clf_model.h5")
classes = ['Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'plastic_bags']


path = "/mnt/vol_1/MLflow/Waste_clf/WasteImagesDataset/Mixed/mixed_741.jpg"

img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

# print(predictions[0])
# plt.imshow(img)
for cls,prob in zip(classes,predictions[0]):

    print('{:<22} {:<10}'.format(cls, np.round(prob*100,2)))
print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")
