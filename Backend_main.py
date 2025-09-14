import os
import shutil
import yaml
from typing import List
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException

# CHANGED: Import the pipeline class and config loader, not the old function
from stage2.main_pipeline import FaceClusteringPipeline
from stage3.score_pipeline import WorkflowOrchestrator
from utils.logger import logger, load_config
from utils.handler import handle_exception

# --- FastAPI App Instantiation ---
app = FastAPI(
    title="Face Clustering API",
    description="An API to upload event images and run a powerful clustering pipeline.",
    version="1.0.0" # Version updated to reflect major change
)

# --- API Endpoints ---

@app.post("/upload", tags=["Image Handling"])
@handle_exception
async def upload_files(images: List[UploadFile] = File(...)):
    """
    Handles the user uploading a list of image files.
    Saves files to the location specified in config.yaml.
    """
    config = load_config()
    upload_folder = config['io']['image_folder']
    os.makedirs(upload_folder, exist_ok=True) # Ensure folder exists

    logger.info(f"Received request to upload {len(images)} files.")
    saved_count = 0
    for image in images:
        if not image.filename:
            logger.warning("Skipping an upload entry with no filename.")
            continue

        save_path = os.path.join(upload_folder, image.filename)
        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            saved_count += 1
            logger.info(f"Successfully saved '{image.filename}' to '{upload_folder}'.")
        finally:
            await image.close()

    logger.info(f"Upload process finished. Saved {saved_count} of {len(images)} images.")
    if saved_count == 0 and images:
        raise HTTPException(status_code=400, detail="No valid files could be saved from the upload request.")
    
    return {"message": f"Successfully uploaded {saved_count} of {len(images)} images."}


@app.post("/process", tags=["Processing"])
@handle_exception
async def process_images(background_tasks: BackgroundTasks):
    """
    Triggers the full, object-oriented face clustering pipeline to run as a background task.
    """
    logger.info("Received request to start image processing.")

    # CHANGED: The background task now runs our entire OOP pipeline
    def run_pipeline_in_background():
        """Wrapper to load config and run the full pipeline."""
        try:
            logger.info("Background task started: Instantiating and running the FaceClusteringPipeline.")
            config = load_config()
            pipeline = FaceClusteringPipeline(config=config)
            pipeline.run()
            logger.info("Background task finished: Pipeline completed successfully.")
        except Exception:
            logger.error("Background task failed: A critical exception occurred in the pipeline.", exc_info=True)

    background_tasks.add_task(run_pipeline_in_background)
    
    logger.info("Face clustering pipeline has been added to the background queue.")
    return {"message": "Processing has been successfully started in the background."}

@app.post("/filter", tags=["Scoring Pipeline"])
@handle_exception
async def trigger_scoring_pipeline(background_tasks: BackgroundTasks):
    
    logger.info("Received API request to start Stage 3: Scoring.")

    def run_scoring_in_background():
        """
        A wrapper function to load the configuration and run our existing
        WorkflowOrchestrator. This is what the background task will execute.
        """
        try:
            logger.info("Background task started: Instantiating and running the WorkflowOrchestrator.")
            config = load_config()
            # Here, we reuse our robust, standalone orchestrator
            orchestrator = WorkflowOrchestrator(config)
            orchestrator.run()
            logger.info("Background task finished: Scoring completed successfully.")
        except Exception:
            logger.error("Background task failed: A critical exception occurred in the scoring pipeline.", exc_info=True)
    
    # Schedule the task to run in the background
    background_tasks.add_task(run_scoring_in_background)
    
    # Immediately return a response to the user
    return {"message": "Accepted. The Stage 3 scoring pipeline has been started in the background."}
