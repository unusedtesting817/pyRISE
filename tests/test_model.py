import pytest
from pyrise2 import load_model

def test_load_model_from_string():
    yaml_string = """
    model:
        name: Test Model
    """
    model = load_model(yaml_string)
    assert model["model"]["name"] == "Test Model"

def test_load_model_from_file(tmp_path):
    yaml_content = """
    model:
        name: Test Model from File
    """
    p = tmp_path / "test_model.yaml"
    p.write_text(yaml_content)
    model = load_model(str(p))
    assert model["model"]["name"] == "Test Model from File"
