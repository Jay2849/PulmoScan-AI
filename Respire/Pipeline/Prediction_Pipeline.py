import os
import io
import base64
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
from tensorflow.keras.models import load_model


class PredictionPipeline:
    def __init__(self, filename="inputImage.jpg"):
        self.filename = filename
        self.model = None
        self.model_path = os.path.join("Artifacts", "Model_Training", "Trained_Model.h5")
        self._load_and_warmup_model()

    def _load_and_warmup_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"Loading model from {self.model_path}...")
                self.model = load_model(self.model_path, compile=False)
                # Warmup: Run dummy inference once to pre-compile execution graph in memory
                dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = self.model(dummy, training=False)
                print("Model loaded & warmed up successfully!")
            except Exception as e:
                print("Initial model load warning:", str(e))

    def predict(self, image_b64=None):
        if self.model is None:
            if os.path.exists(self.model_path):
                self._load_and_warmup_model()
            else:
                return [{"image": "Error: Model file not found on server."}]

        try:
            # Process image in-memory if base64 provided, else fallback to file
            if image_b64 and len(image_b64) > 50:
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif os.path.exists(self.filename):
                img = Image.open(self.filename).convert("RGB")
            else:
                return [{"image": "Error: Invalid image source."}]

            img = img.resize((224, 224))
            test_image = np.array(img, dtype=np.float32) / 255.0
            test_image = np.expand_dims(test_image, axis=0)

            # Direct tensor call is 3x to 5x faster than model.predict() in Flask
            preds = self.model(test_image, training=False)
            result = np.argmax(preds.numpy(), axis=1)
            print("Prediction result index:", result)

            if result[0] == 1:
                prediction = 'Normal'
            else:
                prediction = 'Adenocarcinoma Cancer'

            return [{"image": prediction}]
        except Exception as e:
            print("Prediction error:", str(e))
            return [{"image": f"Prediction Error: {str(e)}"}]