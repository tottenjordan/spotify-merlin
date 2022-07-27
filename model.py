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

import os
from time import time
from typing import List, Any

import nvtabular as nvt
import nvtabular.ops as ops

# disable INFO and DEBUG logging everywhere
import logging
logging.disable(logging.WARNING)

from nvtabular.ops import (
    Categorify,
    TagAsUserID,
    TagAsItemID,
    TagAsItemFeatures,
    TagAsUserFeatures,
    AddMetadata,
    ListSlice
)

from merlin.schema.tags import Tags
from merlin.io.dataset import Dataset as MerlinDataset
import merlin.models.tf as mm
from merlin.models.utils.example_utils import workflow_fit_transform

import tensorflow as tf

# ============================
#  Params to define

WORKFLOW_DIR = 'XXXXXXXXXX'


# ============================


# define workflow 
workflow = nvt.Workflow.load(f"{WORKFLOW_DIR}")

# define schema
schema = workflow.output_schema

# define embeddings
embeddings = ops.get_embedding_sizes(workflow)

#define

def create_two_tower(
    train_data: List[Any],
    valid_data: str,
)