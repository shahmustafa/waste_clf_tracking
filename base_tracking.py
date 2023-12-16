import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns


# Experiment results
artifacts_dir = "artifacts"
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)


## Experiment deials
exp_name = 'BaseTracking-1'
run_name = 'CLR_C7Run-1'
classification_type = 'Multi_7'
dataset_dir = "/mnt/vol_1/waste_dataset/Multi_7_Class"
train_dir = dataset_dir + '/train'
test_dir = dataset_dir + '/test'
model_file = artifacts_dir + '/' + classification_type + exp_name + run_name + "waste_clf_model.h5"





## Hyper-parameters
epochs = 30
batch_size = 96
valid_split = 0.1

## LOAD DATA

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, validation_split=valid_split,
                                                                    subset="training",
                                                                    seed=42, batch_size=batch_size, smart_resize=True,
                                                                    image_size=(256, 256))
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, validation_split=valid_split,
                                                                   subset="validation",
                                                                   seed=42, batch_size=batch_size, smart_resize=True,
                                                                   image_size=(256, 256))

classes = train_dataset.class_names
numClasses = len(train_dataset.class_names)
print(classes)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

## Model Training

baseModel = tf.keras.applications.MobileNetV3Large(input_shape=(256, 256, 3), weights='imagenet', include_top=False,
                                                   classes=numClasses)
for layers in baseModel.layers[:-6]:
    layers.trainable = False

last_output = baseModel.layers[-1].output
x = tf.keras.layers.Dropout(0.45)(last_output)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation=tf.keras.activations.elu, kernel_regularizer=tf.keras.regularizers.l1(0.045),
                          activity_regularizer=tf.keras.regularizers.l1(0.045), kernel_initializer='he_normal')(x)
x = tf.keras.layers.Dropout(0.45)(x)
x = tf.keras.layers.Dense(numClasses, activation='softmax')(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=x)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

lrCallback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))
stepDecay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.1 * 0.1 ** math.floor(epoch / 6))
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[lrCallback, stepDecay])

# Model save
model.save(model_file)

## Model Evaluation

# create 2 subplots
fig, ax = plt.subplots(nrows=2, ncols=1,  figsize=(12, 7))

ax[0].plot(range(0, epochs), history.history["loss"], color="b", label="Train Loss")
ax[0].plot(range(0, epochs), history.history["val_loss"], color="r", label="Val Loss")
# ax[0].set_title("Training-Validation Loss")
# ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(range(0, epochs), history.history["accuracy"], color="b", label="Train Accuracy")
ax[1].plot(range(0, epochs), history.history["val_accuracy"], color="r", label="Val Accuracy")
# ax[1].set_title("Training-Validation Accuracy")
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()


# plot 2 subplots
fig.suptitle('Loss and Accuracy')
plt.savefig(artifacts_dir+'/LossAndAccuracy.png')

train_loss, train_acc = model.evaluate(train_dataset, verbose=0)
val_loss, val_acc = model.evaluate(test_dataset, verbose=0)

# Confusion matrix
def plot_confusion_matrix(cm, target_names, cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}%; misclass={:0.4f}%'.format(accuracy, misclass))
    # plt.show()
    plt.savefig(artifacts_dir + '/ConfusionMatrix.png')


true = []
predictions = []


for i in os.listdir(test_dir):
  folderPath = os.path.join(test_dir, i)
  for j in os.listdir(folderPath)[:550]:
    fullPath = os.path.join(folderPath, j)
    try:
      img = tf.keras.preprocessing.image.load_img(fullPath, target_size=(256, 256))
      img_array = tf.keras.preprocessing.image.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0)

      preds = model.predict(img_array)
      true.append(classes.index(i))
      predictions.append(np.argmax(preds))
    except:
      print("Error on image:", fullPath)

plot_confusion_matrix(tf.math.confusion_matrix(true, predictions), classes)

clr = classification_report(true, predictions, target_names=classes, digits=2, output_dict=True)
print("Classification Report:\n----------------------\n", clr)
plt.figure(figsize=(10, 8))
plt.title('Classification Report')
ax = sns.heatmap(pd.DataFrame(clr).iloc[:-1, :].T, annot=True)
plt.savefig(artifacts_dir + '/ClassificationReport.png')

## MLflow Logging

test_data = np.vstack([x for x, _ in test_dataset])

# model_signature = infer_signature(test_data, model.predict(test_data))

mlflow.set_experiment(exp_name)
# mlflow.log_artifact(dataset_dir)
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("learning_rate", lrCallback)
    mlflow.set_tag("Classification", classification_type)
    mlflow.set_tag("Classes", classes)
    mlflow.set_tag("Dataset", dataset_dir)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_artifact(artifacts_dir+ '/ClassificationReport.png')
    mlflow.log_artifact(artifacts_dir+'/LossAndAccuracy.png')
    mlflow.log_artifact(artifacts_dir + '/ConfusionMatrix.png')
    mlflow.log_artifact(model_file)

mlflow.end_run()