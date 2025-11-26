"""File utilities for bioinformatics agent."""
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import mimetypes
import json
import gzip
import zipfile
import tarfile


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File information
    """
    path = Path(file_path)
    
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "error": "File not found"
        }
    
    stat = path.stat()
    
    info = {
        "exists": True,
        "path": str(path),
        "name": path.name,
        "size": stat.st_size,
        "size_human": format_bytes(stat.st_size),
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "suffix": path.suffix,
        "format": detect_file_format(str(path)).value if path.is_file() else None,
    }
    
    # Add mime type
    if path.is_file():
        mime_type, _ = mimetypes.guess_type(str(path))
        info["mime_type"] = mime_type
    
    # Check if compressed
    info["is_compressed"] = path.suffix.lower() in ['.gz', '.zip', '.tar', '.bz2', '.xz']
    
    return info


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def calculate_checksum(file_path: str, algorithm: str = "md5") -> str:
    """Calculate file checksum.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def read_json(file_path: str) -> Dict[str, Any]:
    """Read JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(data: Dict[str, Any], file_path: str, indent: int = 2):
    """Write JSON file.
    
    Args:
        data: Data to write
        file_path: Output path
        indent: Indentation level
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Parsed YAML data
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(data: Dict[str, Any], file_path: str):
    """Write YAML file.
    
    Args:
        data: Data to write
        file_path: Output path
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def find_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """Find files matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob)
        recursive: Search recursively
        
    Returns:
        List of matching files
    """
    path = Path(directory)
    
    if recursive:
        return list(path.rglob(pattern))
    else:
        return list(path.glob(pattern))


def copy_with_metadata(src: str, dst: str):
    """Copy file preserving metadata.
    
    Args:
        src: Source file
        dst: Destination
    """
    shutil.copy2(src, dst)


def extract_archive(
    archive_path: str,
    extract_to: Optional[str] = None
) -> str:
    """Extract compressed archive.
    
    Args:
        archive_path: Path to archive
        extract_to: Extraction directory
        
    Returns:
        Extraction directory
    """
    path = Path(archive_path)
    
    if extract_to is None:
        extract_to = path.parent / path.stem
    
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.gz' and path.stem.endswith('.tar'):
        # tar.gz file
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_path)
    elif path.suffix == '.tar':
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(extract_path)
    elif path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif path.suffix == '.gz':
        # Single gzipped file
        output_file = extract_path / path.stem
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"Unsupported archive format: {path.suffix}")
    
    return str(extract_path)


def create_archive(
    source_dir: str,
    output_path: str,
    format: str = "zip"
) -> str:
    """Create compressed archive.
    
    Args:
        source_dir: Directory to compress
        output_path: Output archive path
        format: Archive format (zip, tar, tar.gz)
        
    Returns:
        Path to created archive
    """
    source = Path(source_dir)
    output = Path(output_path)
    
    if format == "zip":
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in source.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(source)
                    zipf.write(file, arcname)
    
    elif format == "tar":
        with tarfile.open(output, 'w') as tar:
            tar.add(source, arcname=source.name)
    
    elif format == "tar.gz":
        with tarfile.open(output, 'w:gz') as tar:
            tar.add(source, arcname=source.name)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(output)


def safe_file_name(name: str) -> str:
    """Convert string to safe filename.
    
    Args:
        name: Original name
        
    Returns:
        Safe filename
    """
    # Remove/replace unsafe characters
    safe_chars = "-_.() "
    safe_name = "".join(c for c in name if c.isalnum() or c in safe_chars)
    
    # Replace spaces with underscores
    safe_name = safe_name.replace(" ", "_")
    
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip(". ")
    
    # Ensure not empty
    if not safe_name:
        safe_name = "unnamed"
    
    return safe_name


def get_unique_path(base_path: Path) -> Path:
    """Get unique path by adding number suffix if needed.
    
    Args:
        base_path: Base path
        
    Returns:
        Unique path
    """
    if not base_path.exists():
        return base_path
    
    # Try adding numbers
    counter = 1
    while True:
        if base_path.is_file():
            new_path = base_path.parent / f"{base_path.stem}_{counter}{base_path.suffix}"
        else:
            new_path = base_path.parent / f"{base_path.name}_{counter}"
        
        if not new_path.exists():
            return new_path
        
        counter += 1