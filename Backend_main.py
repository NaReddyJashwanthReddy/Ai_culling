import os
import shutil
import json
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import the core logic and utilities from our other modules
from stage2.main_pipeline import FaceClusteringPipeline
from stage3.score_pipeline import WorkflowOrchestrator
from stage3.scoring_engine import PhotoScorer
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
        config = load_config()
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

idc={}
with open(config['scoring']['categorical_matrices_output'], 'w') as f:
    json.dump(idc, f) 

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

@app.get("/api/categorical_matrices", tags=["Workflow Results"])
@handle_exception
async def get_categorical_matrices():
    """Fetches the current categorical matrices settings."""
    config = load_config()
    with open(config['scoring']['categorical_matrices_output'], 'r') as f:
        idc = json.load(f)
    return JSONResponse(content=idc)
@app.get("/api/gallery-data", tags=["Workflow Results"])
@handle_exception
async def get_gallery_data(filters: Optional[str] = Query(None)):
    """
    Loads all original images, merges with scoring data, and applies
    server-side filters before returning results for the interactive gallery.
    """

    if APP_STATE["status"] != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Results not yet available. Current status: {APP_STATE['status']}")

    image_folder = config['io']['image_folder']
    results_csv = config['scoring']['output_csv']

    try:
        all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images_df = pd.DataFrame(all_images, columns=['photo_path'])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image folder '{image_folder}' not found.")

    scores_df = pd.DataFrame() # Create an empty DataFrame by default

    if os.path.exists(results_csv):
        try:
            scores_df = pd.read_csv(results_csv)
        except pd.errors.EmptyDataError:
            logger.warning("Results CSV is empty.")
            pass # Keep scores_df as empty

    # Merge dataframes; this will keep all images even if scores_df is empty
    gallery_df = pd.merge(images_df, scores_df, on='photo_path', how='left')
    
    # Fill NaN values for images that were not scored or if the file was empty
    gallery_df['results'].fillna('ccccccccc', inplace=True)
    
    #update the results.

    
    # Server-side filtering logic
    '''if filters and filters.strip():
        try:
            filter_indices = [int(i) for i in filters.split(',')]
            mask = pd.Series([True] * len(gallery_df))
            for index in filter_indices:
                if 0 <= index < 9:
                    mask = mask & (gallery_df['results'].str[index] == 'b')
            gallery_df = gallery_df[mask]
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid filter format.")'''

    gallery_df['image_url'] = '/images/' + gallery_df['photo_path']
    gallery_df.to_csv(results_csv, index=False)
    return JSONResponse(content=json.loads(gallery_df.to_json(orient='records')))

@app.post("/api/update-thresholds", tags=["Workflow"])
@handle_exception
async def update_categorical_matrices(thresholds: Optional[str] = Query(None)):
    """Updates the categorical matrices with new threshold values."""
    config = load_config()
    with open(config['scoring']['categorical_matrices_output'], 'r') as f:
        idc = json.load(f)

    if APP_STATE["status"] != "COMPLETE":
        raise HTTPException(status_code=404, detail=f"Results not yet available. Current status: {APP_STATE['status']}")

    image_folder = config['io']['image_folder']
    results_csv = config['scoring']['output_csv']
    if os.path.exists(results_csv):
        photo_results = pd.read_csv(results_csv)
    else:
        return
    
    if thresholds and thresholds.strip():
        try:
            threshold_values = [float(v) for v in thresholds.split(',')]
            if len(threshold_values) != 3:
                raise ValueError("Exactly three threshold values required.")
            eye_v, smile_v, focus_v = threshold_values
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid threshold format. Provide three comma-separated float values.")

    photo_update=PhotoScorer(config)

    for image in idc.keys():
        for type in idc[image].keys():
            for person in idc[image][type]:
                metrics=person
                metrics_bool = photo_update._calculate_threshold_setting(metrics, eye_v, smile_v, focus_v)
                person.update(metrics_bool)

        scores, _ = photo_update._aggregate_scores(idc[image])

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

        if output_string:
            photo_results.loc[photo_results['photo_path']==image,'results']=output_string
    return JSONResponse(content=json.loads(photo_results.to_json(orient='records')))

@app.post("/api/reset", tags=["Workflow"])
@handle_exception
async def reset_workflow():
    """
    Resets the application to its initial state, clearing all generated data.
    """
    logger.info("Received request to reset the entire workflow.")
    
    # 1. Clear generated directories
    for folder_key in ['image_folder', 'cluster_folder']:
        folder_path = config['io'][folder_key]
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
    # 2. Delete generated files
    for file_key in ['output_csv', 'labels_json']:
        file_path = config['scoring'][file_key]
        if os.path.exists(file_path):
            os.remove(file_path)
            
    # 3. Reset the application state
    set_app_state("IDLE", "System reset. Ready for new upload.")
    

    return {"message": "Workflow has been reset successfully."}
