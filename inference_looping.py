import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
imgs_dir = '/mnt/vol_1/waste_dataset/Mixed_Class/Segregated/plastic'

model = load_model("file:///mnt/vol_1/waste_clf_tracking/mlruns/799454089708736406/1f63f1fe3a47446387b349e9dc84f1f8/artifacts/Multi_3BaseTracking-1C3Run-1waste_clf_model.h5")
classes = ['Organic Waste', 'Paper and Cardboard', 'Plastic']

n = 1
prob = []
for img in os.listdir(imgs_dir):
    # img = cv2.imread(os.path.join(imgs_dir, img))
    img = tf.keras.preprocessing.image.load_img(os.path.join(imgs_dir, img), target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    # print(np.round(predictions[0] * 100, 2))
    prob.append(np.round(predictions[0] * 100, 2))
    # np.round(prob * 100, 2)
df = pd.DataFrame(prob, columns=classes)
df.to_csv('prediction_3class_plastic.csv')




