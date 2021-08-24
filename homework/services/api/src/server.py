import numpy as np
import tensorflow as tf
import tritonclient.http as httpclient
from flask import Flask, request, Response


IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 224, 224, 3
LABELS = ['cat', 'dog', 'horse']

app = Flask(__name__)

def preprocess(data):
    image = tf.io.decode_jpeg(data, channels=CHANNELS)
    image = tf.image.resize_with_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = image / 255

    return image.numpy().reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))

triton_client = httpclient.InferenceServerClient(url='triton:8000')
input_name = 'input_1'
output_name = 'predictions'

@app.route('/predict', methods=['POST'])
def predict():
    inputs = preprocess(request.files['image'].read())
    input1 = httpclient.InferInput(input_name, (inputs.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), 'FP32')
    input1.set_data_from_numpy(inputs, binary_data=False)
    output = httpclient.InferRequestedOutput(output_name,  binary_data=False)
    response = triton_client.infer('classifier', model_version='0', inputs=[input1], outputs=[output])
    scores = response.as_numpy(output_name)
    predictions = np.argmax(scores, 1)

    return Response(response=str(LABELS[predictions[0]]) + '\n', status=200, mimetype="text")


app.run(host="0.0.0.0", port=5000, threaded=False)
