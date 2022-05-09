import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pyautogui
import time
from PIL import Image
import pydirectinput

max_length = 1

# ################################################################################################################
characters = ['d', 'w', 's', 'a']
# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

model = tf.keras.models.load_model('captcha.model')

################################################################################################################
# Inference

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text



def rgb2gray_norm(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]

    if r == 255.0 and g == 255.0 and b == 255.0:
        return 1.0

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray / 255.0

def pil_to_tensor(img: Image):
    arr = np.array(img.transpose(2)).tolist()

    new_arr = []

    for _ in range(len(arr)):
        new_arr.append([])

    for i in new_arr:
        for j in range(len(arr[0])):
            i.append([])

    for i, ar in enumerate(arr):
        for j, ar2 in enumerate(ar):
            new_arr[i][j].append(rgb2gray_norm(ar2))

    return tf.convert_to_tensor(np.array([new_arr]), dtype='float32')


while True:
    time.sleep(0.5)

    left = 920
    top = 963
    right = 1000
    bottom = 1040

    img = pyautogui.screenshot().crop((left, top, right, bottom))
    pixel = img.getpixel((0, 0))
    
    if pixel[0] > 230 and pixel[1] > 230 and pixel[2] > 230:
        if pixel[0] < 255 and pixel[1] < 255 and pixel[2] < 255:
            
            preds = prediction_model.predict(pil_to_tensor(img))
            pred_texts = decode_batch_predictions(preds)
            print(img.getpixel((0, 0)))
            print('bukva: ' + pred_texts[0])
            #pydirectinput.press(pred_texts[0])