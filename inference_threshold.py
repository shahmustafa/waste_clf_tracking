import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = load_model("file:///mnt/vol_1/waste_clf_tracking/mlruns/799454089708736406/a7dbd67c4eea4039a36eb840997c7025/artifacts/Multi_3BaseTracking-1cups_C3Run-1waste_clf_model.h5")
classes = ['Organic Waste', 'Paper and Cardboard', 'Plastic']


path = "/mnt/vol_1/waste_dataset/Mixed_Class/Mixed/mixed_109.jpg"

img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
predictions = np.round(predictions*100,2)

organic = predictions[0][0]
paper = predictions[0][1]
plastic = predictions[0][2]
pp = max(paper,plastic)
difference = abs(organic-pp)
print("Organic: ",organic)
print("Paper: ",paper)
print("Plastic: ",plastic)
print("Difference: ",difference)
print()
print()

if organic > 83 != pp > 92:
    print("Class: Segregated")
elif difference > 70:
    print("Class: Segregated")
elif organic < 8:
    print("Class: Segregated")
elif pp < 8:
    print("Class: Segregated")
else:
    print("Class: Mixed")




# for cls,prob in zip(classes,predictions[0]):
#
#     print('{:<22} {:<10}'.format(cls, np.round(prob*100,2)))
# print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")
