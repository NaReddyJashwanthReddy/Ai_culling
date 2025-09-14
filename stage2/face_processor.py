import cv2
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from retinaface import RetinaFace
from deepface import DeepFace

# CHANGED: Import now uses an absolute path from the project root.
from utils.logger import logger
from utils.handler import handle_exception_sync

class FaceProcessor:
    # ... The class logic is identical to the previous version ...
    MAX_FILE_SIZE_MB = 20
    MIN_IMAGE_RESOLUTION = (100, 100)

    def __init__(self, model_name):
        self.model_name = model_name
        self._validate_model()

    @handle_exception_sync
    def _validate_model(self):
        
        logger.info(f"FaceProcessor is building the '{self.model_name}' model...")
        DeepFace.build_model(self.model_name)
        logger.info("DeepFace model built successfully.")

    @handle_exception_sync
    def process_image(self, filepath):
        filename = os.path.basename(filepath)
        if not self._is_image_valid(filepath, filename):
            return []

        image = cv2.imread(filepath)
        if image is None:
            logger.error(f"OpenCV could not read '{filename}'. Skipping.")
            return []

        detected_faces = RetinaFace.detect_faces(filepath)
        if not isinstance(detected_faces, dict):
            return []

        image_face_data = []
        for face_id, data in detected_faces.items():
            x1, y1, x2, y2 = data['facial_area']
            face_img = image[y1:y2, x1:x2]
            
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                embedding_obj = DeepFace.represent(face_img, model_name=self.model_name, enforce_detection=False)
                image_face_data.append({
                    "filename": filename,
                    "facial_area": data['facial_area'],
                    "embedding": embedding_obj[0]["embedding"]
                })
        return image_face_data
        

    @handle_exception_sync
    def _is_image_valid(self, filepath, filename):
        
        if os.path.getsize(filepath) > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.warning(f"Skipping '{filename}': File size > {self.MAX_FILE_SIZE_MB}MB.")
            return False
        with Image.open(filepath) as img:
            if img.width < self.MIN_IMAGE_RESOLUTION[0] or img.height < self.MIN_IMAGE_RESOLUTION[1]:
                logger.warning(f"Skipping '{filename}': Resolution below minimum.")
                return False
            
        return True
