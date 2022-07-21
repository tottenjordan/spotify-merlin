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
"""KFP components."""

from typing import Optional
from . import config

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output


# =============================================
#            convert_to_parquet_op
# =============================================
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def convert_parquet_op(
    output_dataset: Output[Dataset],
    bucket_name: str,
    # data_path_prefix: str,
    output_path_defined_dir: str,
    data_dir_pattern: str,
    # data_paths: list,
    split: str,
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
    '''
    Component to create NVTabular definition.
    
    Args:
    output_dataset: Output[Dataset]
      Output metadata with references to the converted CSV files in GCS
      and the split name.The path to the files are in GCS fuse format:
      /gcs/<bucket name>/path/to/file
    bucket: gcs bucket holding train & valid data
    data_path_prefix: file path to GCS blobl object (e.g., gs://...data/path/prefix.../blob.xxx)
    data_paths: list
      List of paths to folders or files on GCS.
      For recursive folder search, set the recursive variable to True:
        'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
        'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
        a combination of both.
    split: str
      Split name of the dataset. Example: train or valid
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    recursive: bool
      Recursivelly search for files in path.
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    memory_limit: Optional[int] = 100_000_000_000
    '''
    
    # =========================================================
    #            import packages
    # =========================================================
    import os
    import logging
    from google.cloud import storage
    
    storage_client = storage.Client()

    from task import (
        create_cluster,
        create_parquet_dataset_definition,
        convert_definition_to_parquet,
        # get_criteo_col_dtypes,
    )

    logging.info('Base path in %s', output_dataset.path)
    

    # =========================================================
    #            Define data paths
    # =========================================================
    # logging.info(f'bucket_name: {bucket_name}')
    # logging.info(f'data_path_prefix: {data_path_prefix}')
    logging.info(f'data_dir_pattern: {data_dir_pattern}')

    
#     delimiter = '/'
#     data_paths = []
    
#     data_blobs = storage_client.list_blobs(bucket_name, prefix=f'{data_path_prefix}/', delimiter=delimiter)
#     for blob in data_blobs:
#         if blob.name[-7:] == 'parquet':
#             data_paths.append(f'gs://{bucket_name}/{blob.name}')
#             # data_paths.append(f'/gcs/{bucket_name}/{blob.name}')
        
#     logging.info(f'data_paths: {data_paths[:5]}')

    # Write metadata
    output_dataset.metadata['split'] = split

    logging.info('Creating cluster')
    create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac,
      memory_limit=memory_limit
    )
  
    logging.info(f'Creating dataset definition from: {data_dir_pattern}')
    dataset = create_parquet_dataset_definition(
      data_paths=f'{data_dir_pattern}', # data_paths,
      recursive=recursive,
      # col_dtypes=get_criteo_col_dtypes(),
      frac_size=frac_size
    )
  
    logging.info(f'Converting Definition to Parquet; {output_dataset.uri}')
    logging.info(f'Parquet Definition Output Path: ; {output_path_defined_dir}/{split}')
    convert_definition_to_parquet(
        output_path=f'{output_path_defined_dir}/{split}', # output_dataset.uri,
        dataset=dataset,
        output_files=num_output_files,
        shuffle=shuffle
    )

# =========================================================
#            analyze_dataset_op
# =========================================================
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    output_path_defined_dir: str,
    output_path_analyzed_dir: str,
    n_workers: int,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
    '''
    Component to generate statistics from the dataset.
    
    Args:
    parquet_dataset: List of strings
      Input metadata with references to the train and valid converted
      datasets in GCS and the split name.
    workflow: Output[Artifact]
      Output metadata with the path to the fitted workflow artifacts
      (statistics).
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    '''
    import logging
    import nvtabular as nvt
  
    from task import (
        create_cluster,
        create_nvt_workflow,
    )

    logging.basicConfig(level=logging.INFO)

    create_cluster(
      n_workers=n_workers,
      device_limit_frac=device_limit_frac,
      device_pool_frac=device_pool_frac,
      memory_limit=memory_limit
    )

    # logging.info(f'Creating Parquet dataset:{parquet_dataset.uri}')
    logging.info(f'Creating Parquet dataset output_path_defined_dir: {output_path_defined_dir}/train')
    dataset = nvt.Dataset(
        path_or_source=f'{output_path_defined_dir}/train', #parquet_dataset.uri,
        engine='parquet',
        part_mem_fraction=frac_size,
        suffix='.parquet'
    )

    logging.info('Creating Workflow')
    # Create Workflow
    nvt_workflow = create_nvt_workflow()

    logging.info('Analyzing dataset')
    nvt_workflow = nvt_workflow.fit(dataset)

    logging.info('Saving Workflow')
    nvt_workflow.save(f'{output_path_analyzed_dir}') # workflow.path)
    
# =========================================================
#            transform_dataset_op
# =========================================================
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    output_path_defined_dir: str,
    output_path_transformed_dir: str,
    output_path_analyzed_dir: str,
    split: str,
    num_output_files: int,
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.6,
    device_pool_frac: float = 0.9,
    frac_size: float = 0.10,
    memory_limit: int = 100_000_000_000
):
    """Component to transform a dataset according to the workflow definitions.
    Args:
        workflow: Input[Artifact]
        Input metadata with the path to the fitted_workflow
        parquet_dataset: Input[Dataset]
              Location of the converted dataset in GCS and split name
        transformed_dataset: Output[Dataset]
        Split name of the transformed dataset.
        shuffle: str
            How to shuffle the converted CSV, default to None. Options:
                PER_PARTITION
                PER_WORKER
                FULL
    device_limit_frac: float = 0.6
    device_pool_frac: float = 0.9
    frac_size: float = 0.10
    """
    import os
    import logging
    import nvtabular as nvt
    from merlin.schema import Tags

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    from task import (
        create_cluster,
        save_dataset,
    )
    def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
        """Uploads a file to GCS bucket"""
        client = storage.Client()
        blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
        blob.bucket._client = client
        blob.upload_from_filename(source_file_name)
    
    def _read_blob_gcs(bucket_name, source_blob_name, destination_filename):
        """Downloads a file from GCS to local directory"""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_filename)
        

    logging.basicConfig(level=logging.INFO)

    transformed_dataset.metadata['split'] = split

    logging.info('Creating cluster')
    create_cluster(
        n_workers=n_workers,
        device_limit_frac=device_limit_frac,
        device_pool_frac=device_pool_frac,
        memory_limit=memory_limit
  )

   # logging.info(f'Creating Parquet dataset:gs://{parquet_dataset.uri}')
    logging.info(f'Creating Parquet dataset:{output_path_defined_dir}/{split}')
    dataset = nvt.Dataset(
        path_or_source=f'{output_path_defined_dir}/{split}', #f'gs://{parquet_dataset.uri}',
        engine='parquet',
        part_mem_fraction=frac_size,
        suffix='.parquet'
    )

    logging.info('Loading Workflow')
    nvt_workflow = nvt.Workflow.load(f'{output_path_analyzed_dir}') # workflow.path)

    logging.info('Transforming Dataset')
    trans_dataset = nvt_workflow.transform(dataset)

    logging.info(f'transformed_dataset.uri: {transformed_dataset.uri}')
    logging.info(f'Saving transformed dataset: {output_path_transformed_dir}/{split}')
    save_dataset(
        dataset=trans_dataset,
        output_path=f'{output_path_transformed_dir}/{split}', # transformed_dataset.uri,
        output_files=num_output_files,
        shuffle=shuffle
    )

    # =========================================================
    #        read and upload files
    # =========================================================
    logging.info('Generating file list for training.')
    
    # get loca directory
    _local_directory = os.getcwd()
    
    _bucket_name='spotify-merlin-v1'
    _prefix='nvt-preprocessing-spotify-v09-subset/nvt-processed/train'
    _filename='_file_list.txt'
    _source_blob_name = f'{_prefix}/{_filename}'
    logging.info(f'_source_blob_name: {_source_blob_name}')
    
    _local_destination_filename = f'{_local_directory}/local_file_list.txt'
    logging.info(f'_local_destination_filename: {_local_destination_filename}')
    
    _read_blob_gcs(
        bucket_name=_bucket_name,
        source_blob_name=f'{_source_blob_name}', 
        destination_filename=_local_destination_filename
    )
    
    # write new '/gcs/' file
    new_lines = []
    with open(_local_destination_filename, 'r') as fp:
        lines = fp.readlines()
        new_lines.append(lines[0])
        for line in lines[1:]:
            new_lines.append(line.replace('gs://', '/gcs/'))

    _new_local_filename = f'{_local_directory}/_gcs_file_list.txt'
    logging.info(f'_new_local_filename: {_new_local_filename}')
    
    with open(_new_local_filename, 'w') as fp:
        fp.writelines(new_lines)
        
    _gcs_uri_destination = f'{output_path_transformed_dir}/{split}'
    
    _upload_blob_gcs(
        gcs_uri=_gcs_uri_destination, 
        source_file_name=_new_local_filename, 
        destination_blob_name='_gcs_file_list.txt'
    )

#     logging.info('Generating file list for training.')
#     logging.info(f'output_path_transformed_dir/split: {output_path_transformed_dir}/{split}')
#     file_list = os.path.join(f'{output_path_transformed_dir}/{split}', '_file_list.txt')
#     print(f"file_list: {file_list}")

#     new_lines = []
#     with open(file_list, 'r') as fp:
#         lines = fp.readlines()
#         new_lines.append(lines[0])
#         for line in lines[1:]:
#             new_lines.append(line.replace('gs://', '/gcs/'))

#     gcs_file_list = os.path.join(f'{output_path_transformed_dir}/{split}', f'_gcs_file_list.txt')
#     with open(gcs_file_list, 'w') as fp:
#         fp.writelines(new_lines)

    # =========================================================
    #        Saving cardinalities
    # =========================================================
    logging.info('Saving cardinalities')
    
    cols_schemas = nvt_workflow.output_schema.select_by_tag(Tags.CATEGORICAL)
    cols_names = cols_schemas.column_names

    cards = []
    for c in cols_names:
        col = cols_schemas.get(c)
        cards.append(col.properties['embedding_sizes']['cardinality'])

    transformed_dataset.metadata['cardinalities'] = cards