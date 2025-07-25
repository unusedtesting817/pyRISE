import pytest
import pandas as pd
from pyrise2.reporting import generate_table, generate_plot

def test_generate_table():
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    table = generate_table(data, "Test Table")
    pd.testing.assert_frame_equal(table, data)

def test_generate_plot():
    # This is a simple test to ensure the function runs without errors
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    generate_plot(data, "Test Plot")
