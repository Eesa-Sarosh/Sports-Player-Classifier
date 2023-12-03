from flask import Flask, request, jsonify
import utility

app = Flask(__name__)


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']

    response = jsonify(utility.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
if __name__ == "__main__":
    utility.load_saved_model()
    app.run(port=5000)