
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage as ndi
import keras
import base64
import io

app = Flask(__name__)

MODEL_PATH = 'notebook/models/mnist_cnn.keras'
model = keras.models.load_model(MODEL_PATH)


def preprocess(image):
    """Convert drawing into MNIST format (28x28)."""

    img = ImageOps.invert(image.convert('L'))
    arr = np.array(img)

    ys, xs = np.where(arr > 50)

    if len(ys) == 0:
        return None

    arr = arr[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    h, w = arr.shape
    scale = 20 / max(h, w)

    small = Image.fromarray(arr).resize(
        (max(1, round(w * scale)),
         max(1, round(h * scale))),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new('L', (28, 28), 0)

    canvas.paste(
        small,
        ((28 - small.width) // 2,
         (28 - small.height) // 2)
    )

    a = np.array(canvas).astype('float32')

    if a.sum() / 255 < 60:
        a = ndi.maximum_filter(a, size=2)

    a = np.array(
        Image.fromarray(a.clip(0, 255).astype('uint8'))
        .filter(ImageFilter.GaussianBlur(0.6))
    ).astype('float32')

    if a.max() > 0:
        a = a / a.max() * 255

    cy, cx = ndi.center_of_mass(a)

    if np.isfinite(cy) and np.isfinite(cx):
        a = ndi.shift(a, (14 - cy, 14 - cx), order=1)

    return (a / 255.0).clip(0, 1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.json['image']

        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))

        x = preprocess(image)

        if x is None:
            return jsonify({'error': 'Please draw a digit first!'})

        # CNN prediction
        x = x.reshape(1, 28, 28, 1)

        prob = model.predict(x, verbose=0)[0]

        # Get only the best prediction
        digit = int(np.argmax(prob))

        return jsonify({
            'digit': digit,
            'probability': round(float(prob[digit]) * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)