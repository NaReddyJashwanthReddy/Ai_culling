Automated Photo Curation Pipeline
An end-to-end Python application for analyzing and scoring large sets of event photos. This pipeline automates the tedious process of sifting through hundreds of images to find the best ones by identifying who is in each photo and evaluating the quality of their expressions and focus.

The project is architected in distinct, modular stages, controllable via standalone scripts or a modern FastAPI server.

📋 Table of Contents
Architectural Overview

✨ Key Features

Prerequisites

⚙️ Installation & Setup

🔧 Configuration

🚀 End-to-End Workflow

Step 1: Place Your Images

Step 2: Run Stage 2 (Face Clustering)

Step 3: The Manual Step (Labeling)

Step 4: Run Stage 3 (Photo Scoring)

📂 Outputs Explained

Future Improvements

🏛️ Architectural Overview
The application is built as a two-stage data pipeline. This separation of concerns ensures that each part of the process is modular, testable, and scalable.

Stage 2: Face Clustering

Input: A folder of images (event_images/).

Process: Uses retina-face for face detection and deepface (with ArcFace) to generate mathematical embeddings for each face. It then performs parallel processing to handle all images efficiently. Finally, it uses scikit-learn's Agglomerative Clustering to group all detected faces by person.

Output: A face_data.csv file mapping every detected face to a cluster ID, and a folder of sample images (output_clusters/) for each person.

Stage 3: Photo Scoring

Input: The face_data.csv, a manually created cluster_labels.json, and the original images.

Process: It merges the cluster data with your manual labels to understand who is in each photo and their importance (e.g., 'main', 'important'). It then uses a PhotoScorer class with a dlib landmark detector to calculate quality metrics for each face (eyes open, smiling, in focus). Finally, it aggregates these metrics based on configurable business rules to generate a final score for each photo.

Output: A photo_scores.csv file, sorted with the best photos at the top.

✨ Key Features
Modular & Object-Oriented: Logic is encapsulated in clean classes (FaceClusteringPipeline, PhotoScorer, WorkflowOrchestrator) for clarity and reusability.

High-Performance: Uses concurrent.futures.ProcessPoolExecutor for parallel processing of images, drastically reducing runtime on multi-core machines.

Fully Configurable: A central config.yaml file controls all paths, model names, performance settings, and scoring thresholds. No code changes needed for tuning.

Robust Error Handling: Custom decorators and specific exception handling ensure the application logs errors gracefully without crashing.

API-Driven (Optional): Includes a main_api.py with a FastAPI server to trigger the workflows via REST endpoints, complete with automatic interactive documentation.

Clean, Analyzable Output: Produces structured CSV files that are easy to read, sort, and use for data analysis.

Prerequisites
Python 3.9+

pip (Python package installer)

C++ compiler (required by dlib). On Windows, install "C++ build tools" from the Visual Studio Installer. On macOS/Linux, g++ or clang usually suffices.

🔧 Configuration
All application behavior is controlled by the config.yaml file. Before running, review and adjust the parameters as needed.

io section: Defines the input folder for your event images.

clustering section: Allows you to tune the face clustering algorithm.

scoring section:

dlib_model_path: Make sure this points to your downloaded .dat file.

labels_json: The path to your manual labels file.

thresholds: Fine-tune what the engine considers a "smile" or "eyes open".

rules: Define the business logic (e.g., require_all_eyes_open: true).

🚀 End-to-End Workflow
Here is the step-by-step process to go from a folder of photos to a scored and ranked list.

Step 1: Place Your Images
Create a folder named event_images in the project root (or update the path in config.yaml).

Copy all your .jpg or .png event photos into this folder.

Step 2: Run Stage 2 (Face Clustering)
This process analyzes all images and groups the faces it finds.

Start the API Server:

uvicorn main_api:app --reload

Navigate to the API Docs: Open your browser to http://127.0.0.1:8000/docs.

Trigger the /cluster endpoint: This will start the clustering process in the background. Monitor the console logs for progress.

Wait for Completion: Once the logs show "Clustering completed successfully," you can proceed.

Step 3: The Manual Step (Labeling)
This is the crucial human-in-the-loop step where you provide context.

Review the Clusters: Open the output_clusters/ folder. It will contain one cropped face image for each person identified (e.g., cluster_0.jpg, cluster_1.jpg, etc.).

Create the Labels File: Create a new file named cluster_labels.json in the project root.

Map Clusters to People: For each cluster, add an entry to the JSON file, assigning a name and a category (main, important, or other).

Example cluster_labels.json:

{
  "0":  "main" ,
  "1":  "main" ,
  "2": "important" ,
  "3": "other" 
}

Step 4: Run Stage 3 (Photo Scoring)
This final step scores every photo based on your labels and quality metrics.

Using the API:

Go back to the API docs at http://127.0.0.1:8000/docs.

Find the POST /score endpoint and execute it. The process will run in the background.

Using the Standalone Script (Alternative):
You can also run the scoring pipeline directly from the command line.

python run_scoring.py

Monitor and Wait: Watch the console logs. A tqdm progress bar will show the photos being scored. When the logs show "Workflow complete," the process is finished.

📂 Outputs Explained
face_data.csv: The raw output of Stage 2. Contains one row for every single face detected in every photo, along with its bounding box (facial_area) and assigned cluster ID.

output_clusters/: Contains the representative face image for each cluster ID, used for manual labeling.

cluster_labels.json: The manual file you create to provide names and categories for each cluster.

photo_scores.csv: This is the final result. A CSV file containing one row per photo, sorted with the best photos at the top (final_score = 1). It includes the compact results_string for detailed diagnostics.

🔮 Future Improvements
The modular architecture of this project serves as a strong foundation for future enhancements:

Containerization: The application can be containerized using Docker for consistent, isolated deployments.

Distributed Task Queues: For massive scale, the score_photo_worker can be converted into a Celery task and distributed across multiple machines using a message broker like RabbitMQ or Redis.

Cloud-Native Deployment: The entire pipeline can be adapted to run on cloud services like AWS, using S3 for storage and serverless functions (Lambda) or container orchestration (ECS/EKS) for scalable compute.

Persistent Database: The face embeddings could be stored in a vector database (e.g., Milvus, Pinecone) to create a persistent gallery, allowing for near-instant recognition of previously seen individuals in new photos.