import cv2
import dlib
import gc
import json
import os
from math import hypot
from utils.logger import logger
from utils.handler import handle_exception_sync

class PhotoScorer:
    """
    Encapsulates the logic to score a photograph based on facial metrics.
    This class is modular, configurable, and designed for testability.
    """
    def __init__(self, config):
        """Initializes the scorer with configuration and pre-loads the dlib model."""
        self.config = config
        self.predictor = self._load_dlib_model()

    def _load_dlib_model(self):
        """Loads the dlib facial landmark predictor. Critical for startup."""
        model_path = self.config['scoring']['dlib_model_path']
        try:
            logger.info(f"Loading dlib facial landmark model from: {model_path}")
            return dlib.shape_predictor(model_path)
        except dlib.error as e:
            logger.error(f"FATAL: dlib model at '{model_path}' is corrupted or invalid.", exc_info=True)
            raise e

    def score_photo(self, image_path, people_in_photo_df):
        """Scores a single photo, returning a structured dictionary."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"final_score": 0, "reason": f"Could not read image"}
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            return {"final_score": 0, "reason": f"OpenCV error: {e}"}

        if not (self._is_sharp(gray_image) > self.config['thresholds']['global_sharpness']):
            return {"final_score": 0, "reason": "Image is too blurry."}

        category_metrics,category_metrics_bool = self._calculate_all_person_metrics(gray_image, people_in_photo_df)
        
        with open(self.config['scoring']['categorical_matrices_output'], 'w') as f:
            json.dump(category_metrics, f, indent=4)

        del image, gray_image
        gc.collect()
        
        #return self._aggregate_scores(category_metrics)
        score, length = self._aggregate_scores(category_metrics_bool)
        print()
        return score, length

    @handle_exception_sync
    def _calculate_all_person_metrics(self, gray_image, people_df):
        """Iterates through people and calculates their individual metrics."""
        category_metrics = {'main': [], 'important': [], 'other': []}
        category_metrics_bool = {'main': [], 'important': [], 'other': []}  
        for _, person in people_df.iterrows():
            # Input Validation
            facial_area_str = person.get('facial_area')
            
            facial_area = [int(p) for p in facial_area_str.strip('[]').split(',')]
            if len(facial_area) != 4: continue

            thresholds = self.config['thresholds']

            metrics = self._calculate_single_person_metrics(gray_image, facial_area)
            metrics_bool = self._calculate_threshold_setting(metrics, thresholds['eye_aspect_ratio'], thresholds['smile_ratio'], thresholds['face_focus'])

            if metrics:
                category = person.get('category', 'other')
                category_metrics.setdefault(category, []).append(metrics)
                category_metrics_bool.setdefault(category, []).append(metrics_bool)

        with open(self.config['scoring']['categorical_matrices_output'], 'r') as f:
            categorical_metrics_list = json.load(f)

        categorical_metrics_list[people_df['filename'].iloc[0]] = category_metrics 

        return categorical_metrics_list, category_metrics_bool

    @handle_exception_sync 
    def _calculate_single_person_metrics(self, gray_image, facial_area):
        """Calculates eye, smile, and focus scores for one person."""
        
        x1, y1, x2, y2 = facial_area
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        landmarks = self.predictor(gray_image, dlib_rect)

        eye = self._get_eye_aspect_ratio(landmarks)
        smile = self._get_smile_score(landmarks)

        face_crop = gray_image[y1:y2, x1:x2]
        focus_score = self._is_sharp(face_crop)
        
        return {"eye": eye, "smile": smile, "focus": focus_score}

    @handle_exception_sync
    def _calculate_threshold_setting(self,matrix,eye_v,smile_v,focus_v):
        """Calculates the threshold setting for eye, smile, and focus."""
        
        eye_score = 1 if matrix['eye'] > eye_v else 0
        smile_score = 1 if matrix['smile'] > smile_v else 0
        focus_score = 1 if matrix['focus'] > focus_v else 0

        return {"eye": eye_score, "smile": smile_score, "focus": focus_score}

    @handle_exception_sync
    def _aggregate_scores(self, category_metrics):
        """Applies configurable business rules to calculate the final photo score."""
        rules = self.config['rules']
        scores = {}
        category_length={}

        for category in ['main', 'important', 'other']:
            metrics_list = category_metrics.get(category, [])
            category_length[category]=len(metrics_list)
            if not metrics_list:
                scores[category] = {"eye": "N/A", "smile": "N/A", "focus": "N/A"}
                continue
            
            eye_op = all if rules['require_all_eyes_open'] else any
            smile_op = all if rules['require_all_smiles'] else any
            focus_op = all if rules['require_all_in_focus'] else any

            eyes = int(eye_op(m['eye'] for m in metrics_list))
            smile = int(smile_op(m['smile'] for m in metrics_list))
            focus = int(focus_op(m['focus'] for m in metrics_list))

            scores[category]= {
                "eyes":eyes,
                "smile":smile,
                "focus":focus
            }
            
        return scores, category_length
            

    def _is_sharp(self, image_gray):
        if image_gray is None or image_gray.size == 0: return 0
        return cv2.Laplacian(image_gray, cv2.CV_64F).var() 

    def _get_eye_aspect_ratio(self, landmarks):
        left_ear = self._calculate_ear([36, 37, 38, 39, 40, 41], landmarks)
        right_ear = self._calculate_ear([42, 43, 44, 45, 46, 47], landmarks)
        return (left_ear + right_ear) / 2.0

    def _calculate_ear(self, eye_points, landmarks):
        p1 = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
        p2 = (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y)
        p3 = (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y)
        p4 = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
        p5 = (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y)
        p6 = (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)
        vert_dist1 = hypot(p2[0] - p6[0], p2[1] - p6[1])
        vert_dist2 = hypot(p3[0] - p5[0], p3[1] - p5[1])
        horiz_dist = hypot(p1[0] - p4[0], p1[1] - p4[1])
        return 0.0 if horiz_dist == 0 else (vert_dist1 + vert_dist2) / (2.0 * horiz_dist)

    def _get_smile_score(self, landmarks):
        p_left = (landmarks.part(48).x, landmarks.part(48).y)
        p_right = (landmarks.part(54).x, landmarks.part(54).y)
        p_jaw_left = (landmarks.part(2).x, landmarks.part(2).y)
        p_jaw_right = (landmarks.part(14).x, landmarks.part(14).y)
        mouth_width = hypot(p_left[0] - p_right[0], p_left[1] - p_right[1])
        jaw_width = hypot(p_jaw_left[0] - p_jaw_right[0], p_jaw_left[1] - p_jaw_right[1])
        return 0.0 if jaw_width == 0 else mouth_width / jaw_width