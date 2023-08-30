import yaml

to_config = "config.yaml"
with open(to_config, "r") as f:
    config = yaml.safe_load(f)