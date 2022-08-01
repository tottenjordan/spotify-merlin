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
""" TwoTower model trainier """

from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags
import merlin.models.tf as mm
from merlin.io.dataset import Dataset as MerlinDataset

import nvtabular.ops as ops

# ========================================
#  Helper functions
# ========================================

def set_job_dirs():
    '''
    Sets job directories based on env variables set by Vertex AI.
    '''

    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_MODEL_DIR)
    if model_dir[0:5] == 'gs://':
        model_dir = model_dir.replace('gs://', '/gcs/')
    checkpoint_dir = os.getenv('AIP_CHECKPOINT_DIR', LOCAL_CHECKPOINT_DIR)
    if checkpoint_dir[0:5] == 'gs://':
        checkpoint_dir = checkpoint_dir.replace('gs://', '/gcs/')

    return model_dir, checkpoint_dir

# ========================================
#  Load processed data to Merlin Dataset
# ========================================








# ========================================
#  trainer
# ========================================

def main(args):
    """Runs a training loop."""

    repeat_dataset = False if args.num_epochs > 0 else True
    model_dir, snapshot_dir = set_job_dirs()
    num_gpus = sum([len(gpus) for gpus in args.gpus])
    batch_size = num_gpus * args.per_gpu_batch_size
    
    # load Workflow and Schema
    workflow = nvt.Workflow.load(f"{WORKFLOW_DIR}")
    schema = workflow.output_schema
    # embeddings = ops.get_embedding_sizes(workflow)
    
    # create Merlin datasets
    train = MerlinDataset(args.TRANSFORMED_TRAIN_DIR + "/*.parquet", schema=schema, part_size="500MB")
    valid = MerlinDataset(args.TRANSFORMED_VALID_DIR + "/*.parquet", schema=schema, part_size="500MB")
    
    # create schema
    # schema = train.schema
    
    
# ENV VARs to Define
WORKFLOW_DIR = 'gs://spotify-merlin-v1/nvt-preprocessing-spotify-v10-subset/nvt-analyzed'
TRANSFORMED_TRAIN_DIR = 'XXXXXX'
TRANSFORMED_TRAIN_DIR = 'XXXXXX'