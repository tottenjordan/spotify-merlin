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

from typing import List, Any
import hugectr
import merlin.models.tf as mm

def create_model(
    train_data: List[Any],
    valid_data: str,
    slot_size_array: List[Any],
    gpus: List[Any],
    max_eval_batches: int = 300,
    batchsize: int = 2048,
    lr: float = 0.001,
    dropout_rate: float = 0.5,
    workspace_size_per_gpu: float = 8000,
    num_dense_features: int = 13,
    num_sparse_features: int = 26,
    nnz_per_slot: int = 2,
    num_workers: int = 12,
    repeat_dataset: bool = True,
):
    if not gpus:
        gpus = [[0]]
    
    solver = hugectr.CreateSolver(
        max_eval_batches=max_eval_batches,
        batchsize_eval=batchsize,
        batchsize=batchsize,
        lr=lr,
        vvgpu=gpus,
        repeat_dataset=repeat_dataset,
        i64_input_key=True
    )
    
    
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Parquet,
        source=train_data,
        eval_source=valid_data,
        slot_size_array=slot_size_array,
        check_type=hugectr.Check_t.Non,
        num_workers=num_workers
    )
    
    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam,
                                     update_type=hugectr.Update_t.Global,
                                     beta1=0.9,
                                     beta2=0.999,
                                     epsilon=0.0000001)