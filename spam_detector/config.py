import tomli
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)
