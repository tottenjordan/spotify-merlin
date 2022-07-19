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
VERTEX_SA = os.getenv("VERTEX_SA",
                      f"vertex-sa@{PROJECT_ID}.iam.gserviceaccount.com")

VERSION = os.getenv("VERSION", "")

MODEL_DISPLAY_NAME = os.getenv("XXXXX", "")

WORKSPACE = os.getenv("WORKSPACE", "")
NVT_IMAGE_URI = os.getenv("NVT_IMAGE_URI", "")
PREPROCESS_PARQUET_PIPELINE_NAME = os.getenv("PREPROCESS_PARQUET_PIPELINE_NAME", "")
PREPROCESS_PARQUET_PIPELINE_ROOT = os.getenv("PREPROCESS_PARQUET_PIPELINE_ROOT", "")
DOCKERNAME = os.getenv("DOCKERNAME", "")

GPU_LIMIT = os.getenv("GPU_LIMIT", "")
GPU_TYPE = os.getenv("GPU_TYPE", "")
CPU_LIMIT = os.getenv("CPU_LIMIT", "")
MEMORY_LIMIT = os.getenv("MEMORY_LIMIT", "")


