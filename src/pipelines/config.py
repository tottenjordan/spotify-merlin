# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vertex pipeline configurations."""

import os

PROJECT_ID = os.getenv("PROJECT_ID", "")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("BUCKET", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
VERTEX_SA = os.getenv("VERTEX_SA",
                      f"vertex-sa@{PROJECT_ID}.iam.gserviceaccount.com")

VERSION = os.getenv("VERSION", "")

MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "")

WORKSPACE = os.getenv("WORKSPACE", "")
NVT_IMAGE_URI = os.getenv("NVT_IMAGE_URI", "")
PREPROCESS_PARQUET_PIPELINE_NAME = os.getenv("PREPROCESS_PARQUET_PIPELINE_NAME", "")
PREPROCESS_PARQUET_PIPELINE_ROOT = os.getenv("PREPROCESS_PARQUET_PIPELINE_ROOT", "")
DOCKERNAME = os.getenv("DOCKERNAME", "")

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "n1-highmem-64")
CPU_LIMIT = os.getenv("CPU_LIMIT", "64")
MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "624G")
GPU_LIMIT = os.getenv("GPU_LIMIT", "4")
GPU_TYPE = os.getenv("GPU_TYPE", "NVIDIA_TESLA_T4")

# train & valid parquet files
# TRAIN_DIR_PARQUET = os.getenv("TRAIN_DIR_PARQUET", "gs://spotify-builtin-2t/train_data_parquet/0000000000**.snappy.parquet")
# VALID_DIR_PARQUET = os.getenv("VALID_DIR_PARQUET", "gs://spotify-builtin-2t/validation_data_parquet/00000000000*.snappy.parquet")


