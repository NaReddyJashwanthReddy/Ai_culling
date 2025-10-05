import argparse
import json
import os
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import the 'tool' (PhotoScorer) and our custom utilities
from stage3.scoring_engine import PhotoScorer
from utils.logger import logger
from utils.handler import handle_exception_sync

# For Parallel Processing: This worker function runs in a separate process.
# This is a robust pattern that avoids issues with object pickling.
def score_photo_worker(photo_group, config):
    """
    A standalone worker function for parallel execution. It instantiates
    a PhotoScorer and scores one photo group.
    """
    filename, people_df = photo_group
    photo_results=[]
    # Architectural Note: For a distributed system (e.g., Celery/SQS), this
    # worker function would be the body of the task.
    scorer = PhotoScorer(config)
    image_folder = config['io']['image_folder']
    image_path = os.path.join(image_folder, filename)

    scores, length = scorer.score_photo(image_path, people_df)

    output_string=''
    if scores:
        for metrix in scores.values():
            for val in metrix.values():
                if val == 0:
                    value_string='a'
                elif val == 1:
                    value_string='b'
                else:
                    value_string='c'
                output_string+=value_string

    if scores:
        photo_results.append({
            'photo_path': filename, 
            'main': length['main'],
            'important':length['important'],
            'others':length['other'],
            'total' : length['main']+length['important']+length['other'],
            'results':output_string
        })
    return photo_results[0]

class WorkflowOrchestrator:
    """
    Orchestrates the entire Stage 3 scoring pipeline, from data loading
    to parallel processing and saving results.
    """
    def __init__(self, config):
        self.config = config
        self.scorer_config = config['scoring']
        self.io_config = config['io']

    def run(self):
        """Executes the full scoring workflow."""
        logger.info("--- Starting Stage 3: Photo Scoring Pipeline ---")
        
        merged_df = self._load_and_prepare_data()
        if merged_df is None:
            logger.error("Data loading failed. Aborting pipeline.")
            return

        # Group data by photo, ready for processing
        photos_to_score = list(merged_df.groupby('filename'))
        
        all_scores = self._run_scoring_in_parallel(photos_to_score)

        if not all_scores:
            logger.warning("No photos were successfully scored.")
            return

        self._save_results(all_scores)
        logger.info(f"✅ Workflow complete. Scores saved to '{self.scorer_config['output_csv']}'")
    

    def _load_and_prepare_data(self):
        """
        Loads, validates, and merges face data with manual labels using
        efficient Pandas operations.
        
        Input/Output Abstraction Note: To use cloud storage, this is the
        only method you would need to change (e.g., replace pd.read_csv
        with a function that reads from an S3 bucket).
        """
        try:
            face_data_csv = self.scorer_config['face_data_csv']
            labels_json = self.scorer_config['labels_json']
            face_data_df = pd.read_csv(face_data_csv)
            with open(labels_json, 'r') as f:
                labels_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load input data: {e}", exc_info=True)
            return None

        labels_list = [{'cluster': int(k), 'category' : v} for k, v in labels_data.items()]
        labels_df = pd.DataFrame(labels_list)
        
        # Data Validation
        if 'cluster' not in labels_df.columns or 'category' not in labels_df.columns:
            logger.error("labels_json is malformed. Must contain 'cluster' and 'category'.")
            return None

        # This single merge is a highly optimized replacement for the old loops
        return pd.merge(face_data_df, labels_df, on='cluster')

    def _run_scoring_in_parallel(self, photos_to_score):
        """Manages the parallel execution of the photo scoring."""
        all_scores = []
        # Use the number of workers from config, or default to all CPU cores
        workers = self.scorer_config.get('parallel_workers') or os.cpu_count()
        logger.info(f"Starting parallel scoring with {workers} worker processes.")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Dispatch all tasks to the process pool
            futures = [executor.submit(score_photo_worker, group, self.config) for group in photos_to_score]
            
            # Process results as they complete, with a progress bar
            for future in tqdm(as_completed(futures), total=len(photos_to_score)):
                result = future.result()
                if result:
                    all_scores.append(result)
        return all_scores

    def _save_results(self, all_scores):
        """Creates a clean, flat DataFrame and saves it to a CSV file."""
        scores_df = pd.DataFrame(all_scores)
        # Idempotency Note: For restartable pipelines, you could add logic here
        # to check if the output file exists and merge/overwrite accordingly.
        scores_df.to_csv(self.scorer_config['output_csv'], index=False)
