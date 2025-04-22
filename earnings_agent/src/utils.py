import os
import json
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime

def ensure_directory_exists(directory: Union[str, Path]) -> None:
    """
    Ensure that a directory exists. If it doesn't, create it.
    
    Args:
        directory: Path to the directory
    """
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a JSON file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the JSON data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Write a dictionary to a JSON file.
    
    Args:
        data: Dictionary to write
        file_path: Path to the JSON file
    """
    file_path = Path(file_path)
    ensure_directory_exists(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def format_currency(value: float) -> str:
    """
    Format a number as a currency string.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted currency string
    """
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """
    Format a number as a percentage string.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%"

def get_current_timestamp() -> str:
    """
    Get the current timestamp in a human-readable format.
    
    Returns:
        Current timestamp as a string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def slugify(text: str) -> str:
    """
    Convert a string into a slug-friendly format.
    
    Args:
        text: Input string
        
    Returns:
        Slugified string
    """
    return text.lower().replace(" ", "-").replace("_", "-")

def validate_file_extension(file_path: Union[str, Path], valid_extensions: List[str]) -> bool:
    """
    Validate if a file has one of the allowed extensions.
    
    Args:
        file_path: Path to the file
        valid_extensions: List of valid extensions (e.g., ['.json', '.txt'])
        
    Returns:
        True if the file has a valid extension, False otherwise
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() in valid_extensions

def load_env_variable(key: str, default: Any = None) -> Any:
    """
    Load an environment variable, returning a default value if not set.
    
    Args:
        key: Environment variable key
        default: Default value to return if the variable is not set
        
    Returns:
        Value of the environment variable or the default
    """
    return os.getenv(key, default)

def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into smaller chunks of a specified size.
    
    Args:
        data: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def log_message(message: str, level: str = "INFO") -> None:
    """
    Log a message with a timestamp and level.
    
    Args:
        message: Message to log
        level: Log level (e.g., "INFO", "ERROR")
    """
    timestamp = get_current_timestamp()
    print(f"[{timestamp}] [{level}] {message}")

def is_valid_json(data: str) -> bool:
    """
    Check if a string is valid JSON.
    
    Args:
        data: String to check
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False
    


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Size of the file in bytes
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return os.path.getsize(file_path)

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension of a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File extension (e.g., ".json")
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() if file_path.exists() else ""   

def get_file_name(file_path: Union[str, Path]) -> str:
    """
    Get the file name without the extension.        

    Args:
        file_path: Path to the file 
    returns:
        File name without the extension
    """
    file_path = Path(file_path)
    return file_path.stem if file_path.exists() else "" 


