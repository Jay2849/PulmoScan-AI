import os
import traceback
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, render_template
from Respire.Pipeline.Prediction_Pipeline import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


# Global instance ensures availability under WSGI/Gunicorn and direct execution
clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(force=True)
        image = data.get('image', '')
        result = clApp.classifier.predict(image_b64=image)
        return jsonify(result)
    except Exception as e:
        print("Prediction route exception:", str(e))
        traceback.print_exc()
        return jsonify([{"image": f"Prediction Error: {str(e)}"}])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)