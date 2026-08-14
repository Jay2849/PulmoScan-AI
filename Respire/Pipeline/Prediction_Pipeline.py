import os
import numpy as np
# pyrefly: ignore [missing-import]
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
# pyrefly: ignore [missing-import]
from Respire.Pipeline.Training_Pipeline.Data_Ingestion import DataIngestionTrainingPipeline
# pyrefly: ignore [missing-import]
from Respire.Pipeline.Training_Pipeline.Base_Model import PrepareBaseModelTrainingPipeline
# pyrefly: ignore [missing-import]
from Respire.Pipeline.Training_Pipeline.Model_Trainer import ModelTrainingPipeline


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = None
        self.model_path = os.path.join("Artifacts", "Model_Training", "Trained_Model.h5")
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path, compile=False)
            except Exception as e:
                print("Initial model load warning:", e)
    
    def predict(self):
        if self.model is None or not os.path.exists(self.model_path):
            print("Model file not found. Initializing pipeline to train model...")
            try:
                ingestion = DataIngestionTrainingPipeline()
                ingestion.main()
                base_model = PrepareBaseModelTrainingPipeline()
                base_model.main()
                trainer = ModelTrainingPipeline()
                trainer.main()
            except Exception as e:
                print("Model generation error:", e)
            self.model = load_model(self.model_path, compile=False)

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(self.model.predict(test_image), axis=1)
        print("Prediction result index:", result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{"image": prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{"image": prediction}]