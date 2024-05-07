import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

with open("model.yaml", "r") as yaml_file:
    loaded_model_yaml = yaml_file.read()
model = tf.keras.models.model_from_json(loaded_model_yaml)
model.load_weights("model.h5")

# Función para realizar la predicción
def predict_image(img):
    # Preprocesamiento de la imagen
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalización

    # Predicción
    prediction = model.predict(np.expand_dims(img, axis=0))
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_label = emotion_labels[np.argmax(prediction)]

    return predicted_label


# Interfaz de Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs="image", 
    outputs="text",
    #interpretation="default",
    title="Emotion Recognition",
    description="Draw or upload an image and get the predicted emotion.",
    #allow_screenshot=True,
    live=True
)

# Ejecutar la interfaz
iface.launch(share=True)