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
"""DeepFM Network in HugeCTR."""
from typing import List, Any
# import hugectr
from mpi4py import MPI

from merlin.models.utils.example_utils import workflow_fit_transform
from merlin.schema.tags import Tags
import merlin.models.tf as mm
from merlin.io.dataset import Dataset as MerlinDataset
import nvtabular.ops as ops

def create_two_tower(
    train_dir: str,
    valid_dir: str,
    workflow_dir: str,
    # slot_size_array: List[Any],
    gpus: List[Any],
    layer_sizes: List[Any] = [1024,512,256],
    # max_eval_batches: int = 300,
    # batchsize: int = 2048,
    # lr: float = 0.001,
    # dropout_rate: float = 0.5,
    # workspace_size_per_gpu: float = 8000,
    # num_dense_features: int = 13,
    # num_sparse_features: int = 26,
    # nnz_per_slot: int = 2,
    # num_workers: int = 12,
    # repeat_dataset: bool = True,
):


    # if not gpus:
    #     gpus = [[0]]
        
    workflow = nvt.Workflow.load(workflow_dir) # gs://spotify-merlin-v1/nvt-preprocessing-spotify-v24/nvt-analyzed
    
    schema = workflow.output_schema
    embeddings = ops.get_embedding_sizes(workflow)
        
    train_data = MerlinDataset(train_dir + "/*.parquet", part_size="500MB")
    valid_data = MerlinDataset(valid_dir + "/*.parquet", part_size="500MB")

    # =========================================================
    #             remove sequence features # TODO: parameterize
    # =========================================================
    two_t_schema = schema.select_by_tag([Tags.ITEM_ID, Tags.ITEM, Tags.USER, Tags.USER_ID])
    two_t_schema_seq = schema.select_by_tag([Tags.SEQUENCE])
    non_seq_col_names = list(set(two_t_schema.column_names) - set(two_t_schema_seq.column_names))
    two_t_schema = two_t_schema[non_seq_col_names]
        
    model = mm.TwoTowerModel(
        two_t_schema,
        query_tower=mm.MLPBlock(layer_sizes, no_activation_last_layer=True),
        item_tower=mm.MLPBlock(layer_sizes, no_activation_last_layer=True),
        samplers=[mm.InBatchSampler()],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )
    
    model.compile(optimizer="adam", run_eagerly=False, metrics=[mm.RecallAt(1), mm.RecallAt(10), mm.NDCGAt(10)])
    
    return model