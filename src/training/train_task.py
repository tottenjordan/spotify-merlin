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

import hugectr
# from hugectr.inference import CreateInferenceSession
# from hugectr.inference import InferenceParams
import hypertune
from two_tower_model import create_two_tower
import utils

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

    parameters_path = os.path.join(model_dir, model_name)
    logging.info('Saving model parameters to: %s', parameters_path)
    model.save_params_to_files(prefix=parameters_path)

    graph_path = os.path.join(model_dir, f'{model_name}.json')
    logging.info('Saving model graph to: %s', graph_path)
    model.graph_to_json(graph_config_file=graph_path)
    
    keras_path = os.path.join(model_dir, '2tower')
    model.save(keras_path)
    
    query_tower = model.retrieval_block.query_block()
    query_tower_path = os.path.join(model_dir, 'query-tower')
    query_tower.save(query_tower_path)
    

    
def main(args):
    """Runs a training loop."""

    repeat_dataset = False if args.num_epochs > 0 else True
    model_dir, snapshot_dir = set_job_dirs()
    num_gpus = sum([len(gpus) for gpus in args.gpus])
    batch_size = num_gpus * args.per_gpu_batch_size
    
    train_data = MerlinDataset(args.train_dir + "/*.parquet", part_size="500MB")
    valid_data = MerlinDataset(args.valid_dir + "/*.parquet", part_size="500MB")

    model = create_two_tower(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        workflow_dir=args.workflow_dir,
        layer_sizes=args.layer_sizes,
        gpus=args.gpus
    )
    
    
    model.summary()

    logging.info('Starting model training')
    model.fit(
        train_data, validation_data=valid_data, batch_size=batch_size, epochs=args.num_epochs
    )


    logging.info('Saving model')
    save_model(model, args.model_name, model_dir)

    # logging.info('Starting model evaluation using %s batches ...', args.eval_batches)
    # metric_value = evaluate_model(model_name=args.model_name,
    #                             model_dir=model_dir,
    #                             eval_data_source=args.valid_data,
    #                             num_batches=args.eval_batches,
    #                             device_id=0,
    #                             max_batchsize=args.per_gpu_batch_size,
    #                             slot_size_array=args.slot_size_array)
    
    # logging.info('%s on the evaluation dataset: %s',HYPERTUNE_METRIC_NAME, metric_value)

    # # Report AUC to Vertex hypertuner
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
        type=int,
        required=True,
        help='layer_sizes'
    )
    # parser.add_argument(
    #     '--num_epochs',
    #     type=int,
    #     required=True,
    #     help='num_epochs
    # )
    
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