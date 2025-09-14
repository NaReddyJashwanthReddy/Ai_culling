import os
import shutil
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

# CHANGED: Imports now use absolute paths from the project root.
from stage2.face_processor import FaceProcessor
from utils.logger import logger
from utils.handler import handle_exception_sync

# This is a new helper function needed because class methods can't be
# easily 'pickled' for multiprocessing by some OSes. A standalone function is more robust.
@handle_exception_sync
def process_image_worker(filepath, config):
    """
    Standalone function for multiprocessing. Creates a FaceProcessor 
    instance within the worker process to handle one image.
    """
    processor = FaceProcessor(model_name=config['model_performance']['model_name'])
    return processor.process_image(filepath)

class FaceClusteringPipeline:
    # ... The class logic is identical to the previous version ...
    def __init__(self, config):
        self.config = config
        self.all_faces_data = []

    def run(self):
        logger.info("--- Starting Face Clustering Pipeline ---")
        self._setup_directories()
        image_files = self._gather_image_files()

        if not image_files:
            return
        
        print(image_files)

        self._process_images_parallel(image_files)

        if not self.all_faces_data:
            logger.warning("No faces were detected. Pipeline finished.")
            return

        self._save_results()
        logger.info("--- Pipeline Complete ---")

    def _setup_directories(self):
        cluster_folder = self.config['io']['cluster_folder']
        if os.path.exists(cluster_folder):
            shutil.rmtree(cluster_folder)
        os.makedirs(cluster_folder, exist_ok=True)

    @handle_exception_sync
    def _gather_image_files(self):
        image_folder = self.config['io']['image_folder']
        return [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        

    def _process_images_parallel(self, image_files):
        workers = self.config['model_performance']['workers'] or os.cpu_count()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # We now call the standalone worker function
            task = partial(process_image_worker, config=self.config)
            futures = {executor.submit(task, fp): fp for fp in image_files}
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    self.all_faces_data.extend(result)
                logger.info(f"Progress: Processed {i}/{len(image_files)} images...")
    
    def _cluster_embeddings(self, df):
        logger.info(f"Clustering {len(df)} faces...")
        embeddings = np.array(df['embedding'].tolist())
        cfg = self.config['clustering']
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric=cfg['affinity'],
            linkage=cfg['linkage'],
            distance_threshold=cfg['threshold']
        )
        df['cluster'] = clustering.fit_predict(embeddings)
        return df

    def _save_results(self):
        logger.info(f"Total faces detected: {len(self.all_faces_data)}")
        df = pd.DataFrame(self.all_faces_data)
        
        clustered_df = self._cluster_embeddings(df)
        
        logger.info(f"Clustering complete. Found {clustered_df['cluster'].nunique()} unique people.")
        clustered_df.to_csv(self.config['io']['output_csv'], index=False)
        logger.info(f"Full face data saved to '{self.config['io']['output_csv']}'")
        
        self._save_representative_faces(clustered_df)

    def _save_representative_faces(self, df):
        logger.info("Saving representative faces for each cluster...")
        image_folder = self.config['io']['image_folder']
        cluster_folder = self.config['io']['cluster_folder']
        affinity = self.config['clustering']['affinity']

        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1: continue
            
            cluster_df = df[df['cluster'] == cluster_id].copy()
            cluster_embeddings = np.array(cluster_df['embedding'].tolist())
            
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cdist(cluster_embeddings, centroid.reshape(1, -1), metric=affinity)
            
            rep_face_idx = distances.argmin()
            representative_face = cluster_df.iloc[rep_face_idx]
            
            img_path = os.path.join(image_folder, representative_face['filename'])
            img = cv2.imread(img_path)
            
            if img is None: continue

            x1, y1, x2, y2 = representative_face['facial_area']
            face_crop = img[y1:y2, x1:x2]
            
            cv2.imwrite(os.path.join(cluster_folder, f'cluster_{cluster_id}.jpg'), face_crop)