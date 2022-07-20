import argparse
import logging
import os
import sys
import time

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import fsspec
import nvtabular as nvt
from merlin.io.shuffle import Shuffle
from nvtabular.ops import Categorify
from nvtabular.ops import Clip
from nvtabular.ops import FillMissing
from nvtabular.ops import Normalize
from nvtabular.utils import device_mem_size

import numpy as np
from typing import Dict, List, Union

# jj packs
import nvtabular.ops as ops
from merlin.schema.tags import Tags

from nvtabular.ops import (
    # Categorify,
    TagAsUserID,
    TagAsItemID,
    TagAsItemFeatures,
    TagAsUserFeatures,
    AddMetadata,
    ListSlice
)

# =============================================
#            TODO: parameterize 
# =============================================
item_features_cat = [
  'artist_name_can',
  'track_name_can',
  'artist_genres_can',
]

item_features_cont = [
  'duration_ms_can',
  'track_pop_can',
  'artist_pop_can',
  'artist_followers_can',
]

playlist_features_cat = [
  'artist_name_seed_track',
  'artist_uri_seed_track',
  'track_name_seed_track',
  'track_uri_seed_track',
  'album_name_seed_track',
  'album_uri_seed_track',
  'artist_genres_seed_track',
  'description_pl',
  'name',
  'collaborative',
]

playlist_features_cont = [
  'duration_seed_track',
  'track_pop_seed_track',
  'artist_pop_seed_track',
  'artist_followers_seed_track',
  'duration_ms_seed_pl',
  'n_songs_pl',
  'num_artists_pl',
  'num_albums_pl',
]
CAT = playlist_features_cat + item_features_cat
CONT = item_features_cont + playlist_features_cont
# =========================================

def create_cluster(
    n_workers,
    device_limit_frac,
    device_pool_frac,
    memory_limit
):
  """Create a Dask cluster to apply the transformations steps to the Dataset."""
  device_size = device_mem_size()
  device_limit = int(device_limit_frac * device_size)
  device_pool_size = int(device_pool_frac * device_size)
  rmm_pool_size = (device_pool_size // 256) * 256

  cluster = LocalCUDACluster(
      n_workers=n_workers,
      device_memory_limit=device_limit,
      rmm_pool_size=rmm_pool_size,
      memory_limit=memory_limit
  )

  return Client(cluster)


# =============================================
#            Create & Save dataset
# =============================================

# def create_parquet_nvt_dataset(data_dir, frac_size):
#   return nvt.Dataset(f'{data_dir}', engine='parquet', part_mem_fraction=frac_size)
def create_parquet_nvt_dataset(
    data_path,
    frac_size
):
  """Create a nvt.Dataset definition for the parquet files."""
  fs = fsspec.filesystem('gs')
  file_list = fs.glob(
      os.path.join(data_path, '*.parquet')
  )

  if not file_list:
    raise FileNotFoundError('Parquet file(s) not found')

  file_list = [os.path.join('gs://', i) for i in file_list]

  return nvt.Dataset(
      file_list,
      engine='parquet',
      part_mem_fraction=frac_size
  )

def save_dataset(
    dataset,
    output_path,
    output_files,
    categorical_cols,
    continuous_cols,
    shuffle=None,
):
  """Save dataset to parquet files to path."""
  
  dict_dtypes = {}
  for col in categorical_cols:
    dict_dtypes[col] = np.int64

  for col in continuous_cols:
    dict_dtypes[col] = np.float32

  dataset.to_parquet(
      output_path=output_path,
      shuffle=shuffle,
      output_files=output_files,
      dtypes=dict_dtypes,
      cats=categorical_cols,
      conts=continuous_cols,
  )
# =============================================

# =============================================
#            Workflow
# =============================================
def create_nvt_workflow():
  '''
  Create a nvt.Workflow definition with transformation all the steps
  '''
  MAX_PADDING = 375
  item_id = ["track_uri_can"] >> Categorify(dtype="int32") >> ops.TagAsItemID() >> ops.AddMetadata(tags=["user_item"])
#   playlist_id = ["pid_pos_id"] >> Categorify(dtype="int32") >> TagAsUserID() 
  
  item_features_cat = [
    'artist_name_can',
    'track_name_can',
    'artist_genres_can',
  ]

  item_features_cont = [
    'duration_ms_can',
    'track_pop_can',
    'artist_pop_can',
    'artist_followers_can',
  ]

  playlist_features_cat = [
    'artist_name_seed_track',
    'artist_uri_seed_track',
    'track_name_seed_track',
    'track_uri_seed_track',
    'album_name_seed_track',
    'album_uri_seed_track',
    'artist_genres_seed_track',
    'description_pl',
    'name',
    'collaborative',
  ]

  playlist_features_cont = [
    'duration_seed_track',
    'track_pop_seed_track',
    'artist_pop_seed_track',
    'artist_followers_seed_track',
    'duration_ms_seed_pl',
    'n_songs_pl',
    'num_artists_pl',
    'num_albums_pl',
  ]

  # subset of features to be tagged
  seq_feats_cont = [
    'duration_ms_songs_pl',
    'artist_pop_pl',
    'artists_followers_pl',
    'track_pop_pl',
  ]

  seq_feats_cat = [
    'artist_name_pl',
    # 'track_uri_pl',
    'track_name_pl',
    'album_name_pl',
    'artist_genres_pl',
    # 'pid_pos_id', 
    # 'pos_pl'
  ]
  
  CAT = playlist_features_cat + item_features_cat
  CONT = item_features_cont + playlist_features_cont
  
  item_feature_cat_node = item_features_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsItemFeatures()

  item_feature_cont_node =  item_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsItemFeatures()

  playlist_feature_cat_node = playlist_features_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> TagAsUserFeatures() 

  playlist_feature_cont_node = playlist_features_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsUserFeatures()

  playlist_feature_cat_seq_node = seq_feats_cat >> nvt.ops.FillMissing() >> Categorify(dtype="int32") >> ListSlice(MAX_PADDING, pad=True, pad_value=0) >> TagAsUserFeatures() >> nvt.ops.AddTags(Tags.SEQUENCE) 

  playlist_feature_cont_seq_node = seq_feats_cont >> nvt.ops.FillMissing() >>  nvt.ops.Normalize() >> TagAsUserFeatures() >> nvt.ops.AddTags(Tags.SEQUENCE)
  
  # define a workflow
  output = item_id \
  + item_feature_cat_node \
  + item_feature_cont_node \
  + playlist_feature_cat_node \
  + playlist_feature_cont_node \
  + playlist_feature_cont_seq_node \
  + playlist_feature_cat_seq_node \
  # playlist_id \

  workflow = nvt.Workflow(output)
  
  return workflow

# =============================================
#            Create Parquet Dataset 
# =============================================

def create_parquet_dataset_definition(
    data_paths,
    recursive,
    # col_dtypes,
    frac_size,
    # sep='\t'
):
  """Create nvt.Dataset definition for Parquet files."""
  fs_spec = fsspec.filesystem('gs')
  rec_symbol = '**' if recursive else '*'

  valid_paths = []
  for path in data_paths:
    try:
      if fs_spec.isfile(path):
        valid_paths.append(path)
      else:
        path = os.path.join(path, rec_symbol)
        for i in fs_spec.glob(path):
          if fs_spec.isfile(i):
            valid_paths.append(f'gs://{i}')
    except FileNotFoundError as fnf_expt:
      print(fnf_expt)
      print('Incorrect path: {path}.')
    except OSError as os_err:
      print(os_err)
      print('Verify access to the bucket.')

  return nvt.Dataset(
      path_or_source=valid_paths,
      engine='parquet',
      # names=list(col_dtypes.keys()),
      # sep=sep,
      # dtypes=col_dtypes,
      part_mem_fraction=frac_size,
      # assume_missing=True
  )

def convert_definition_to_parquet(
    output_path,
    dataset,
    output_files,
    shuffle=None
):
  """Convert Parquet files to parquet and write to GCS."""
  if shuffle == 'None':
    shuffle = None
  else:
    try:
      shuffle = getattr(Shuffle, shuffle)
    except:
      print('Shuffle method not available. Using default.')
      shuffle = None

  dataset.to_parquet(
      output_path,
      shuffle=shuffle,
      output_files=output_files
  )

# =============================================
#            Create nv-tabular definition
# =============================================
def main_convert(args):
  logging.info('Creating cluster')
  client = create_cluster(
    args.n_workers,
    args.device_limit_frac,
    args.device_pool_frac,
    args.memory_limit
  )
  logging.info('Creating parquet dataset definition')
  dataset = create_parquet_dataset_definition(
    data_paths=args.parq_data_path, 
    # args.sep,
    recursive=False, 
    # get_criteo_col_dtypes(), 
    frac_size=args.frac_size
  )

  logging.info('Converting definition to Parquet')
  convert_definition_to_parquet(
    args.output_path,
    dataset,
    args.output_files
  )
# =============================================
#            Analyse Dataset 
# =============================================
def main_analyze(args):
  logging.info('Creating cluster')
  client = create_cluster(
    args.n_workers,
    args.device_limit_frac,
    args.device_pool_frac,
    args.memory_limit
  )
  
  logging.info('Creating Parquet dataset')
  dataset = create_parquet_nvt_dataset(
    data_dir=args.parquet_data_path,
    frac_size=args.frac_size
  )
  
  logging.info('Creating Workflow')
  # Create Workflow
  nvt_workflow = create_nvt_workflow()
  
  logging.info('Analyzing dataset')
  nvt_workflow = nvt_workflow.fit(dataset)

  logging.info('Saving Workflow')
  nvt_workflow.save(args.output_path)
# =============================================

# =============================================
#            Transform Dataset 
# =============================================
def main_transform(args):
  client = create_cluster(
      args.n_workers,
      args.device_limit_frac,
      args.device_pool_frac,
      args.memory_limit
  )

  # nvt_workflow = create_nvt_workflow()
  nvt_workflow = nvt.Workflow.load(args.workflow_path, client)

  dataset = create_parquet_nvt_dataset(args.parquet_data_path, frac_size=args.frac_size)

  logging.info('Transforming Dataset')
  transformed_dataset = nvt_workflow.transform(dataset)

  logging.info('Saving transformed dataset')
  save_dataset(
      transformed_dataset,
      output_path=args.output_path,
      output_files=args.output_files,
      categorical_cols=CAT,
      continuous_cols=CONT,
      shuffle=nvt.io.Shuffle.PER_PARTITION,
  )

  # =============================================
  #            args
  # =============================================
def parse_args():
  """Parses command line arguments."""

  parser = argparse.ArgumentParser()
  
  parser.add_argument('--task',
                      type=str,
                      required=False)
  parser.add_argument('--parquet_data_path',
                      type=str,
                      required=False)
  parser.add_argument('--parq_data_path',
                      required=False,
                      nargs='+')
  parser.add_argument('--output_path',
                      type=str,
                      required=False)
  parser.add_argument('--output_files',
                      type=int,
                      required=False)
  parser.add_argument('--workflow_path',
                      type=str,
                      required=False)
  parser.add_argument('--n_workers',
                      type=int,
                      required=False)
  parser.add_argument('--frac_size',
                      type=float,
                      required=False,
                      default=0.10)
  parser.add_argument('--memory_limit',
                      type=int,
                      required=False,
                      default=100_000_000_000)
  parser.add_argument('--device_limit_frac',
                      type=float,
                      required=False,
                      default=0.60)
  parser.add_argument('--device_pool_frac',
                      type=float,
                      required=False,
                      default=0.90)

  return parser.parse_args()
  

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s - %(message)s',
                      level=logging.INFO, 
                      datefmt='%d-%m-%y %H:%M:%S',
                      stream=sys.stdout)

  parsed_args = parse_args()

  start_time = time.time()
  logging.info('Timing task')

  if parsed_args.task == 'transform':
    main_transform(parsed_args)
  elif parsed_args.task == 'analyze':
    main_analyze(parsed_args)
  elif parsed_args.task == 'convert':
    main_convert(parsed_args)

  end_time = time.time()
  elapsed_time = end_time - start_time
  logging.info('Task completed. Elapsed time: %s', elapsed_time)