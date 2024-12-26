from abc import ABC 
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import pathlib
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE=1000000

class PqDAO:

    def __init__(self, dataset_path:  Union[str, pathlib.PosixPath]=None, echo_schema=False):
        self.dataset_path = dataset_path
        self.dataset = self._init_dataset_if_exists(dataset_path)
        if echo_schema:
            print(self.dataset.schema)

    def _init_dataset_if_exists(self, dataset_path):
        if dataset_path is not None:
            return ds.dataset(dataset_path, format="parquet", partitioning="hive")
            # self.dataset = pd.ParquetDataset(dataset_path, use_legacy_dataset=False)

    def stream_read(self, columns=None, filter=None, batch_size=_DEFAULT_BATCH_SIZE):
        """Reads the parquet dataset in batches of 'batch_size' number of rows (default=1,000,000)
        then yields each batch for further work downstream.
        """
        if columns is not None:
            for batch in self.dataset.to_batches(columns=columns, filter=filter, batch_size=batch_size):
                yield batch
        else:
            for batch in self.dataset.to_batches(filter=filter, batch_size=batch_size):
                yield batch

    def stream_write(self, dataset_path, columns=None, partition=[], filter=None, batch_size=_DEFAULT_BATCH_SIZE):
        if columns is not None:
            for batch in self.dataset.to_batches(filter=filter, batch_size=batch_size):
                self.write(dataset_path, batch, partition=partition)
        else:
            for batch in self.dataset.to_batches(columns=columns, filter=filter, batch_size=batch_size):
                self.write(dataset_path, batch, partition=partition)

    def build_expression(self):
        """Helper for building filter expressions
        """
        pass

    def read(self): 
        pass

    def stream_from_pandas(self, dataframe_chunk, dataset_path, partition=[], schema=None, overwrite=False):
        """
        """
        try:
            if schema:
                rb = pa.RecordBatch.from_pandas(dataframe_chunk, schema=schema)
                self.write(dataset_path, dataset=rb, partition=partition, schema=schema)
            else:
                rb = pa.RecordBatch.from_pandas(dataframe_chunk)
                self.write(dataset_path, dataset=rb, partition=partition)
        except TypeError:
            size_in_bytes = dataframe_chunk.memory_usage().sum()
            num_chunks = size_in_bytes // 1E9
            if num_chunks == 0:
                raise TypeError("Chunksize is not the issue")
            dataframe_chunks = np.array_split(dataframe_chunk, num_chunks)
            if schema:
                for chunk in dataframe_chunks:
                    rb = pa.RecordBatch.from_pandas(chunk, schema=schema)
                    self.write(dataset_path, dataset=rb, partition=partition, schema=schema)
            else:
                for chunk in dataframe_chunks:
                    rb = pa.RecordBatch.from_pandas(chunk)
                    self.write(dataset_path, dataset=rb, partition=partition)

    def to_pandas(self, filter=None, columns=None):
        """Read the dataset into a pandas dataframe

        Filter should be a dataset column filter. `ds.field('column') == 'criteria'`

        read_to_pandas(ds.field('year')==2021)
        """
        if columns:
            return self.dataset.to_table(columns=columns, filter=filter).to_pandas()
        else:
            return self.dataset.to_table(filter=filter).to_pandas()

    def write(self, dataset_path, dataset=None, partition=[], existing_data_behavior='overwrite_or_ignore', schema=None):
        if dataset is None:
            dataset=self.dataset
        if schema:
            ds.write_dataset(dataset, dataset_path, 
                            partitioning = partition, 
                            basename_template=uuid4().hex+'{i}.parquet', 
                            existing_data_behavior=existing_data_behavior,
                            # max_rows_per_file=nrows, need pyarrow 9.0.0
                            format='parquet',
                            partitioning_flavor='hive',
                            schema=schema)
        else:
            ds.write_dataset(dataset, dataset_path, 
                            partitioning = partition, 
                            basename_template=uuid4().hex+'{i}.parquet', 
                            existing_data_behavior=existing_data_behavior,
                            # max_rows_per_file=nrows, need pyarrow 9.0.0
                            format='parquet',
                            partitioning_flavor='hive')

    def validate_schema(self):
        pass

    def get_fragments(self):
        pass

    def repack(self, dest_ds_path, partition=[], batch_size=_DEFAULT_BATCH_SIZE):
        """Iterate over dataset partitions and compact filesets into 
        files of desired size
        """
        for batch in self.stream_read(batch_size):
            self.write(dest_ds_path, dataset=batch, partition=partition, existing_data_behavior='delete_matching')
    
    def get_unique_values(column:str) -> List[str]:
        unique_values = []
        for chunk in self.stream_read(columns=[column]):
            df = chunk.to_pandas()
            unique_names += df[column].unique().tolist()
        return list(set(unique_values))



    

# class PqDAO(ABC):

#     def __init__(self, dataset_path: Union[str, pathlib.PosixPath], filter:Union[List[List[Tuple]],List[Tuple]]=None):
#         self.dataset_path = dataset_path
#         self.dataset = self._read_dataset(dataset_path)

#     def get_schema(self):
#         return self.dataset.schema

#     def _read_dataset(self, dataset_path=None, filters=None):
#         if dataset_path is None:
#             dataset_path = self.dataset_path
#         return pq.ParquetDataset(dataset_path, filters=filters, use_legacy_dataset=False)

#     def to_pandas(self):
#         return self.dataset.read().to_pandas()

#     def get_fragments(self):
#         return self.dataset.fragments

#     def validate_schema(self):
#         self.dataset.validate_schema


class PqMergeHandler:

    def __init__(self, ds_left, ds_right):
        self.ds_left = ds_left
        self.ds_right = ds_right
        self.df = None

    def merge(self, how='inner', left_on=None, right_on=None):
        df_left = self.ds_left.to_pandas()
        df_right = self.ds_right.to_pandas()
        self.df = pd.merge(df_left, df_right, how=how, left_on=left_on, right_on=right_on)
        return self.df

    def write_dataset(self, ds_path, partition_cols=None):
        self.df.to_parquet(ds_path, partition_cols=partition_cols)
