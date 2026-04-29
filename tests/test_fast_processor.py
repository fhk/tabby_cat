import pytest
import os
import pandas as pd
from tabby_cat.fast_processor import FastProcessor
import graph_assembler_cpp

def test_fast_processor_basic():
    fp = FastProcessor()
    assert fp.assembler is not None

    # Test with empty data
    nodes_df = fp.get_nodes_df()
    edges_df = fp.get_edges_df()
    assert nodes_df.empty
    assert edges_df.empty

def test_assembler_manual_data():
    fp = FastProcessor()
    # Check it raises error for non-existent file as expected from C++
    with pytest.raises(RuntimeError):
        fp.load_data("non_existent.pbf", "non_existent.geojson")

def test_parquet_export(tmpdir):
    fp = FastProcessor()
    output_dir = str(tmpdir.mkdir("output"))
    nodes_df, edges_df = fp.save_to_parquet(output_dir)

    assert os.path.exists(os.path.join(output_dir, 'nodes.parquet'))
    assert os.path.exists(os.path.join(output_dir, 'edges.parquet'))

    # Reload and check
    reloaded_nodes = pd.read_parquet(os.path.join(output_dir, 'nodes.parquet'))
    assert len(reloaded_nodes) == len(nodes_df)
