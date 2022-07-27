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
from merlin.models.utils.dataset import unique_rows_by_features

import nvtabular.ops as ops

import pandas as pd


# ============================
#  Params to define

WORKFLOW_DIR = 'XXXXXXXXXX'
TRAIN_DIR_PATH = 'XXXXXXXXXX'
VALID_DIR_PATH = 'XXXXXXXXXX'
JOB_OUTPUT_DIR = 'XXXXXXXXXX'
# ============================



# ========================================
#  Build Model
# ========================================
def main(args):
    '''
    Runs a training loop
    
    '''

    repeat_dataset = False if args.num_epochs > 0 else True
    model_dir, snapshot_dir = set_job_dirs()
    num_gpus = sum([len(gpus) for gpus in args.gpus])
    batch_size = num_gpus * args.per_gpu_batch_size
    
    # define workflow 
    workflow = nvt.Workflow.load(f"{WORKFLOW_DIR}")
    
    # define schema
    schema = workflow.output_schema
    
    # define embeddings
    embedding_dims = {}
    embeddings = ops.get_embedding_sizes(workflow)
    for k in list(embeddings.keys()):
        embedding_dims.update({k: embeddings[k][1]})
        
    # Remove sequence/ragged features from this example # TODO: revisit
    two_tower_model_schema = schema.select_by_tag(
        [
            Tags.ITEM_ID, 
            Tags.ITEM, 
            Tags.USER, 
            Tags.USER_ID
        ]
    )
    two_tower_model_schema_sequence = schema.select_by_tag([Tags.SEQUENCE])
    non_sequence_cols = list(set(two_tower_model_schema.column_names) - set(two_tower_model_schema_sequence.column_names))
    two_tower_model_schema=two_tower_model_schema[non_sequence_cols]
        
    
    
    train = MerlinDataset(TRAIN_DIR_PATH + "/*.parquet", schema=schema, part_size="500MB")
    valid = MerlinDataset(VALID_DIR_PATH + "/*.parquet", schema=schema, part_size="500MB")
    
    
    model = mm.TwoTowerModel(
        two_t_schema,
        query_tower=mm.MLPBlock(
            [1024,512,256], 
            no_activation_last_layer=True
        ),
        item_tower=mm.MLPBlock(
            [1024,512,256], 
            no_activation_last_layer=True
        ),
        samplers=[
            mm.InBatchSampler()
        ],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )
    
    model.compile(optimizer="adam", run_eagerly=False, metrics=[mm.RecallAt(1), mm.RecallAt(10), mm.NDCGAt(10)])
    
    logging.info('Starting model training')
    model.fit(train, validation_data=valid, batch_size=2048, epochs=args.NUM_EPOCHS)
    
    # Save Query Model
    query_model_artifact_path = f'{JOB_OUTPUT_DIR}/saved-query-model'
    model.save(query_model_artifact_path)
    
    # Save Track Embeddings for Matching Engine
    item_features = (
        unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID)
        .compute()
        .reset_index(drop=True)
    )
    
    item_embs = model.item_embeddings(
        MerlinDataset(item_features, schema=schema), batch_size=1024
    )
    item_embs_df = item_embs.compute(scheduler="synchronous")
    
    # store with track ID only
    drop_cols = schema.select_by_tag([Tags.ITEM]).column_names
    drop_cols.remove('track_uri_can')
    
    item_embeddings = item_embs_df.drop(
        columns=drop_cols
    )
    
    item_data = pd.read_parquet('categories/unique.track_uri_can.parquet')
    
    lookup_dict = dict(item_data['track_uri_can'])
    item_embeddings_pd = item_embeddings.to_pandas()

    item_embeddings_pd['track_uri_can'] = item_embeddings_pd['track_uri_can'].apply(lambda l: lookup_dict[l])
    
    item_embeddings.to_parquet(os.path.join(JOB_OUTPUT_DIR,'track-embeddings'))