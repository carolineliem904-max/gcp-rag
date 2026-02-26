"""
Cloud Storage Service
=====================
Handles uploading and downloading files to/from Google Cloud Storage (GCS).

What is GCS?
  Think of it like Google Drive for your app — a place to store files
  (PDFs, images, etc.) that is accessible from anywhere, including Cloud Run.

Why store PDFs in GCS instead of locally?
  - Cloud Run containers restart and lose local files
  - GCS is persistent, durable, and accessible by all your services
  - It's cheap: ~$0.02/GB/month

Flow in this project:
  User uploads PDF → FastAPI saves it to GCS → Pipeline reads from GCS → Processes it
"""

import os
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# Read bucket name from .env
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")


def get_storage_client() -> storage.Client:
    """
    Create and return a GCS client.
    Authentication is handled automatically via Application Default Credentials (ADC).
    Project is passed explicitly because inside a Docker container there is
    no active gcloud config to infer it from automatically.
    """
    return storage.Client(project=PROJECT_ID)


def upload_file(local_file_path: str, destination_blob_name: str) -> str:
    """
    Upload a local file to Cloud Storage.

    Args:
        local_file_path:      path to the file on your machine
                              e.g. "./sample_docs/sample.txt"
        destination_blob_name: name to give the file inside the bucket
                              e.g. "documents/sample.txt"

    Returns:
        The GCS URI of the uploaded file, e.g. "gs://bucket-name/documents/sample.txt"
    """
    client = get_storage_client()

    # A "bucket" is like a folder at the top level
    bucket = client.bucket(BUCKET_NAME)

    # A "blob" is the actual file object inside the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(local_file_path)

    gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
    print(f"  Uploaded: {local_file_path} → {gcs_uri}")
    return gcs_uri


def download_file(source_blob_name: str, local_destination_path: str) -> str:
    """
    Download a file from Cloud Storage to your local machine (or container).

    Args:
        source_blob_name:       name of the file inside the bucket
                                e.g. "documents/sample.txt"
        local_destination_path: where to save it locally
                                e.g. "/tmp/sample.txt"

    Returns:
        The local path where the file was saved.
    """
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)

    # Make sure the destination directory exists
    os.makedirs(os.path.dirname(local_destination_path), exist_ok=True)

    blob.download_to_filename(local_destination_path)
    print(f"  Downloaded: gs://{BUCKET_NAME}/{source_blob_name} → {local_destination_path}")
    return local_destination_path


def list_files(prefix: str = "documents/") -> list[str]:
    """
    List all files in the bucket under a given prefix (like a folder).

    Args:
        prefix: folder path inside the bucket, e.g. "documents/"

    Returns:
        List of blob names (file paths inside the bucket)
    """
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]