import os
import shutil
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import the core logic and utilities from our other modules
from stage2.main_pipeline import FaceClusteringPipeline
from stage3.score_pipeline import WorkflowOrchestrator
from utils.logger import logger, load_config
from utils.handler import handle_exception

# --- Application State Management ---
# In a real multi-user app, this state would be managed in a database like Redis.
# For this single-user project, a simple global dictionary is clean and effective.
APP_STATE = {
    "status": "IDLE", # Can be IDLE, CLUSTERING, READY_FOR_LABELING, SCORING, COMPLETE, ERROR
    "message": "System is ready. Please upload images."
}

def set_app_state(status: str, message: str):
    """A helper function to safely update the global application state and log the change."""
    APP_STATE["status"] = status
    APP_STATE["message"] = message
    logger.info(f"State changed to {status}: {message}")

# --- Background Task Wrappers ---
# These functions wrap our pipeline classes to manage state during background execution.

def run_clustering_task():
    """Wrapper for the Stage 2 clustering pipeline."""
    try:
        set_app_state("CLUSTERING", "Face clustering is in progress... This may take several minutes.")
        config = load_config()
        pipeline = FaceClusteringPipeline(config=config)
        pipeline.run()
        set_app_state("READY_FOR_LABELING", "Clustering complete. Please label the clusters.")
    except Exception as e:
        set_app_state("ERROR", f"An error occurred during clustering: {e}")
        logger.error("Clustering pipeline failed.", exc_info=True)

def run_scoring_task():
    """Wrapper for the Stage 3 scoring pipeline."""
    try:
        set_app_state("SCORING", "Photo scoring is in progress... This may take a few minutes.")
        config = load_config('config.yaml')
        orchestrator = WorkflowOrchestrator(config)
        orchestrator.run()
        set_app_state("COMPLETE", "Scoring complete. Results are available.")
    except Exception as e:
        set_app_state("ERROR", f"An error occurred during scoring: {e}")
        logger.error("Scoring pipeline failed.", exc_info=True)

# --- FastAPI App Instantiation ---
app = FastAPI(
    title="Automated Photo Curation API",
    description="A complete API for the multi-stage photo clustering and scoring pipeline.",
    version="1.0.0"
)

# --- Serve Frontend and Static Files ---
# This section makes our application self-contained.

@app.get("/", include_in_schema=False)
async def read_index():
    """Serves the main frontend application (index.html)."""
    return FileResponse('index.html')

config = load_config()
# Create directories if they don't exist to prevent errors on first run
os.makedirs(config['io']['cluster_folder'], exist_ok=True)
os.makedirs(config['io']['image_folder'], exist_ok=True)

# Mount static directories to serve images directly to the frontend
app.mount("/clusters", StaticFiles(directory=config['io']['cluster_folder']), name="clusters")
app.mount("/images", StaticFiles(directory=config['io']['image_folder']), name="images")

# --- API Endpoints for Frontend ---

@app.get("/api/status", tags=["Workflow"])
async def get_status():
    """Returns the current status of the backend pipeline for UI polling."""
    return APP_STATE

@app.post("/api/upload", tags=["Workflow"])
@handle_exception
async def upload_files(images: List[UploadFile] = File(...)):
    """Handles uploading all event images and resets the pipeline."""
    upload_folder = config['io']['image_folder']
    
    # Clear previous event data for a fresh run
    for folder in [config['io']['cluster_folder'], upload_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    for image in images:
        save_path = os.path.join(upload_folder, image.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        await image.close()
        
    set_app_state("READY_FOR_CLUSTERING", f"{len(images)} images uploaded. Ready to start clustering.")
    return {"message": APP_STATE["message"]}

@app.post("/api/cluster", tags=["Workflow"])
@handle_exception
async def trigger_clustering(background_tasks: BackgroundTasks):
    """Triggers the Stage 2 face clustering pipeline as a background task."""
    if APP_STATE["status"] != "READY_FOR_CLUSTERING":
        raise HTTPException(status_code=409, detail=f"Cannot start clustering. System status is '{APP_STATE['status']}'. Please upload images first.")
    
    background_tasks.add_task(run_clustering_task)
    set_app_state("CLUSTERING_QUEUED", "Clustering process has been queued.")
    return {"message": "Accepted. Face clustering started in the background."}

@app.get("/api/clusters", tags=["Workflow"])
@handle_exception
async def get_cluster_data():
    """Returns a list of cluster image URLs for the labeling UI."""
    cluster_folder = config['io']['cluster_folder']
    if not os.path.exists(cluster_folder):
        raise HTTPException(status_code=404, detail="Cluster output folder not found. Please run the clustering process first.")
    
    cluster_images = sorted([f for f in os.listdir(cluster_folder) if f.endswith('.jpg')])
    return {"cluster_image_urls": [f"/clusters/{filename}" for filename in cluster_images]}

@app.post("/api/labels", tags=["Workflow"])
@handle_exception
async def upload_cluster_labels(labels_file: UploadFile = File(...)):
    """Receives the manually created cluster_labels.json file from the frontend."""
    labels_path = config['scoring']['labels_json']
    content = await labels_file.read()
    try:
        json.loads(content) # Validate that it's valid JSON
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")
    with open(labels_path, "wb") as f:
        f.write(content)
        
    set_app_state("READY_FOR_SCORING", "Labels uploaded. Ready to start scoring.")
    return {"message": "Labels file uploaded successfully."}

@app.post("/api/score", tags=["Workflow"])
@handle_exception
async def trigger_scoring(background_tasks: BackgroundTasks):
    """Triggers the Stage 3 photo scoring pipeline as a background task."""
    if APP_STATE["status"] != "READY_FOR_SCORING":
        raise HTTPException(status_code=409, detail=f"Cannot start scoring. System status is '{APP_STATE['status']}'. Please upload labels first.")
        
    background_tasks.add_task(run_scoring_task)
    set_app_state("SCORING_QUEUED", "Scoring process has been queued.")
    return {"message": "Accepted. Photo scoring started in the background."}

@app.get("/api/results", tags=["Workflow"])
@handle_exception
async def get_results():
    """Returns the final scored and ranked photos as a JSON array for the results page."""
    if APP_STATE["status"] != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Results are not yet available. Current status: {APP_STATE['status']}")
        
    results_csv = config['scoring']['output_csv']
    try:
        df = pd.read_csv(results_csv)
        # Add a full image URL for the frontend to easily display the final photos
        df['image_url'] = '/images/' + df['filename']
        # Convert DataFrame to a JSON array of records, handling potential NaN values
        results_json = json.loads(df.to_json(orient='records'))
        return JSONResponse(content=results_json)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Results file not found.")