import mlflow.tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os


# Setting an experiment for manual logging
mlflow.set_experiment(experiment_name="Base_tracking")
# Start an MLflow run
with mlflow.start_run():
    #     mlflow.tensorflow.autolog()

    # Data Preprocessing
    # LOAD DATA
    DIR = "/mnt/vol_1/waste_dataset/Multi_Class"
    batch_size = 128
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DIR, validation_split=0.1, subset="training", seed=42,
        batch_size=batch_size, smart_resize=True, image_size=(256, 256)
    )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DIR, validation_split=0.1, subset="validation", seed=42,
        batch_size=batch_size, smart_resize=True, image_size=(256, 256)
    )

    classes = train_dataset.class_names
    numClasses = len(train_dataset.class_names)

    # Model Training
    baseModel = tf.keras.applications.MobileNetV3Large(input_shape=(256, 256, 3), weights='imagenet', include_top=False,
                                                       classes=numClasses)
    for layers in baseModel.layers[:-6]:
        layers.trainable = False

    last_output = baseModel.layers[-1].output
    x = tf.keras.layers.Dropout(0.45)(last_output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=tf.keras.activations.elu,
                              kernel_regularizer=tf.keras.regularizers.l1(0.045),
                              activity_regularizer=tf.keras.regularizers.l1(0.045), kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    x = tf.keras.layers.Dense(numClasses, activation='softmax')(x)

    model = tf.keras.Model(inputs=baseModel.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00125),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    epochs = 20
    lrCallback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))
    stepDecay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.1 * 0.1 ** math.floor(epoch / 6))
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[lrCallback, stepDecay])

    # Model Evaluation
    plt.plot(range(0, epochs), history.history["loss"], color="b", label="Train Loss")
    plt.plot(range(0, epochs), history.history["val_loss"], color="r", label="Val Loss")
    plt.title("Trainig-Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Train_val_loss.png')

    # Log parameters and metrics
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.00125  # Change this to your learning rate
    })

    for epoch in range(epochs):
        mlflow.log_metrics({
            f'training_loss_{epoch}': history.history['loss'][epoch],
            f'training_accuracy_{epoch}': history.history['accuracy'][epoch],
            f'validation_loss_{epoch}': history.history['val_loss'][epoch],
            f'validation_accuracy_{epoch}': history.history['val_accuracy'][epoch]
        })

    # Log the 'Train_val_loss.png' plot as an artifact
    mlflow.log_artifact('Train_val_loss.png')

    # Log the learning rate vs. loss plot as an artifact
    learning_rates = 1e-3 * (10 ** (np.arange(epochs) / 30))
    plt.plot(learning_rates, history.history['loss'], lw=3, color='#48e073')
    plt.title('Learning rate vs. loss')
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.savefig('Learning_rate_loss.png')
    mlflow.log_artifact('Learning_rate_loss.png')
