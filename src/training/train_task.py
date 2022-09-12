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
"""DeepFM Network trainer."""

import argparse
import json
import logging
import os
import sys
import time
import pandas as pd

# import hugectr
# from hugectr.inference import CreateInferenceSession
# from hugectr.inference import InferenceParams
# import hypertune
from two_tower_model import create_two_tower
import utils
import merlin.models.tf as mm
from merlin.io.dataset import Dataset # as MerlinDataset
import nvtabular as nvt
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags

SNAPSHOT_DIR = 'snapshots'
HYPERTUNE_METRIC_NAME = 'AUC'

LOCAL_MODEL_DIR = '/tmp/saved_model'
LOCAL_CHECKPOINT_DIR = '/tmp/checkpoints'


def set_job_dirs():
    """Sets job directories based on env variables set by Vertex AI."""

    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_MODEL_DIR)
    if model_dir[0:5] == 'gs://':
        model_dir = model_dir.replace('gs://', '/gcs/')
    checkpoint_dir = os.getenv('AIP_CHECKPOINT_DIR', LOCAL_CHECKPOINT_DIR)
    if checkpoint_dir[0:5] == 'gs://':
        checkpoint_dir = checkpoint_dir.replace('gs://', '/gcs/')

    return model_dir, checkpoint_dir


def save_model(model, model_name, model_dir):
    """Saves model graph and model parameters."""
    
    query_tower = model.retrieval_block.query_block()
    query_tower_path = os.path.join(model_dir, 'query-tower')
    logging.info('Saving query_tower to query_tower_path: %s', query_tower_path)
    query_tower.save(query_tower_path)
    

    
def main(args):
    """Runs a training loop."""

    repeat_dataset = False if args.num_epochs > 0 else True
    model_dir, snapshot_dir = set_job_dirs()
    logging.info(f'from set_job_dirs():\n model_dir: {model_dir}\n snapshot_dir: {snapshot_dir}')
    
    num_gpus = sum([len(gpus) for gpus in args.gpus])
    logging.info(f'num_gpus: {num_gpus}')
    
    batch_size = num_gpus * args.per_gpu_batch_size
    logging.info(f'batch_size: {batch_size}')
    
    logging.info(f'Loading workflow & schema from : {args.workflow_dir}')
    workflow = nvt.Workflow.load(args.workflow_dir) # gs://{BUCKET}/..../nvt-analyzed
    schema = workflow.output_schema
    
    train_data = Dataset(args.train_dir + "/*.parquet", part_size="500MB")
    valid_data = Dataset(args.valid_dir + "/*.parquet", part_size="500MB")

    model = create_two_tower(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        workflow_dir=args.workflow_dir,
        # layer_sizes=args.layer_sizes,
        gpus=args.gpus
    )
    
    # model.set_retrieval_candidates_for_evaluation(train_data)
    model.compile(
        optimizer="adam", 
        run_eagerly=False, 
        metrics=[mm.RecallAt(1), mm.RecallAt(10), mm.NDCGAt(10)]
    )

    logging.info('Starting model training')
    model.fit(
        train_data, 
        validation_data=valid_data, 
        batch_size=batch_size, 
        epochs=args.num_epochs,
    )
 
    # =============================================
    #        save retrieval (query) tower
    # =============================================
    logging.info(f'Saving model to {args.model_dir}')
    save_model(model, args.model_name, args.model_dir)
    
    # =============================================
    #        save candidate item embeddings
    # =============================================
    '''
    embedding file for Matching Engine Index needs to be saved to empty GCS directory
    '''
    
    item_features = (
        unique_rows_by_features(
            train_data, 
            Tags.ITEM, 
            Tags.ITEM_ID
        )
        .compute()
        .reset_index(drop=True)
    )
    logging.info(f'Shape of item_features {item_features.shape}')
    
    item_embs = model.item_embeddings(
        Dataset(
            item_features, 
            schema=schema
        ),
        batch_size=1024
    )
    
    item_embs_df = item_embs.compute(
        scheduler="synchronous"
    )
    
    drop_cols = schema.select_by_tag([Tags.ITEM]).column_names
    drop_cols.remove('track_uri_can')
    
    logging.info(f'Removing drop columns: {drop_cols}')
    item_embeddings = item_embs_df.drop(columns=drop_cols)
    named_embeddings = item_embeddings.to_pandas()
    
    # TODO: parameterize with pipe output
    CANDIDATE_TRACK_IDS_GCS_URI = 'gs://spotify-merlin-v1/nvt-preprocessing-spotify-v32-subset/nvt-analyzed/categories/unique.track_uri_can.parquet' # TODO: fix
    logging.info(f'Loading unique candidate IDs from: {CANDIDATE_TRACK_IDS_GCS_URI}')
    candidate_track_ids = pd.read_parquet(f'{CANDIDATE_TRACK_IDS_GCS_URI}')
    
    lookup_dict = dict(candidate_track_ids['track_uri_can'])
    named_embeddings['track_uri_can'] = named_embeddings['track_uri_can'].apply(lambda l: lookup_dict[l].encode('utf-8'))
    
    # save as parquet file and CSV/json (for matching engine index)
    named_embeddings.to_parquet(
        os.path.join(
            args.model_dir, 
            "candidate-embeddings.parquet"
        )
    )
    EMB_INDEX_DIR = f'{args.model_dir}/index-dir'
    EMBEDDINGS_CSV_FILENAME = 'candidate-embeddings.csv'
    EMBEDDINGS_CSV_DESTINATION_URI = f'{EMB_INDEX_DIR}/{EMBEDDINGS_CSV_FILENAME}'
    logging.info(f'Saving Candidate Embeddings file to: {EMBEDDINGS_CSV_DESTINATION_URI}')
    named_embeddings.to_csv(
        EMBEDDINGS_CSV_DESTINATION_URI, 
        encoding='utf-8', 
        index=False
    )
    
    # =============================================
    #            Evaluate Model
    # =============================================
    # https://github.com/NVIDIA-Merlin/models/blob/main/tests/unit/tf/models/test_retrieval.py#L184
    
    # metrics = model.evaluate(data, batch_size=10, item_corpus=data, return_dict=True, steps=1)
    
    # logging.info('Starting model evaluation using %s batches ...', args.eval_batches)
    # metric_value = evaluate_model(model_name=args.model_name,
    #                             model_dir=model_dir,
    #                             eval_data_source=args.valid_data,
    #                             num_batches=args.eval_batches,
    #                             device_id=0,
    #                             max_batchsize=args.per_gpu_batch_size,
    #                             slot_size_array=args.slot_size_array)
    
    # logging.info('%s on the evaluation dataset: %s',HYPERTUNE_METRIC_NAME, metric_value)

    
    # =============================================
    #            Hyperparamerter Tuning
    # =============================================
    # # Report AUC to Vertex for hp-tuning
    
    # logging.info('Reporting %s metric at %s to Vertex hypertuner',HYPERTUNE_METRIC_NAME, metric_value)
    # hpt = hypertune.HyperTune()
    # hpt.report_hyperparameter_tuning_metric(
    #     hyperparameter_metric_tag=HYPERTUNE_METRIC_NAME,
    #     metric_value=metric_value,
    #     global_step=args.max_iter if repeat_dataset else args.num_epochs
    # )
    
def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        default='twotower',
        help='Model Name.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        required=True,
        help='Path to training data _file_list.txt'
    )
    parser.add_argument(
        '--valid_dir',
        type=str,
        required=True,
        help='Path to validation data _file_list.txt'
    )
    parser.add_argument(
        '--schema',
        type=str,
        required=True,
        help='Path to the schema.pbtxt file'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        required=True,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True,
        help='num_epochs'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        required=False,
        default='[[0]]',
        help='GPU devices to use for Preprocessing'
    )
    parser.add_argument(
        '--per_gpu_batch_size',
        type=int,
        required=True,
        help='Per GPU Batch size'
    )
    parser.add_argument(
        '--layer_sizes',
        type=str,
        required=False,
        help='layer_sizes'
    )
    parser.add_argument(
        '--workflow_dir',
        type=str,
        required=False,
        help='Path to saved workflow.pkl e.g., nvt-analyzed'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path for saving model artifacts'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )

    parsed_args = parse_args()

    parsed_args.gpus = json.loads(parsed_args.gpus)

    # parsed_args.slot_size_array = [
    #     int(i) for i in parsed_args.slot_size_array.split(sep=' ')
    # ]

    logging.info('Args: %s', parsed_args)
    start_time = time.time()
    logging.info('Starting training')

    main(parsed_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info('Training completed. Elapsed time: %s', elapsed_time )