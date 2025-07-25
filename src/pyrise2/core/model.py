import yaml

def load_model(path_or_yaml):
    """Load DSGE model from file or YAML string."""
    if isinstance(path_or_yaml, str) and (path_or_yaml.endswith(".yaml") or path_or_yaml.endswith(".yml")):
        with open(path_or_yaml, 'r') as f:
            return yaml.safe_load(f)
    else:
        return yaml.safe_load(path_or_yaml)
