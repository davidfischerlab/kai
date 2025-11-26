#!/usr/bin/env python3
"""Download and extract kai retrieval database from Zenodo.

This script downloads the pre-built retrieval database (knowledge base) for kai
from Zenodo and extracts it to the appropriate location in the user's home directory.

Usage:
------
    python scripts/download_retrieval_data.py VERSION [--zenodo-url URL]

Arguments:
    VERSION         Version identifier (e.g., 251121) - REQUIRED
    --zenodo-url    Optional custom Zenodo URL template (uses VERSION if provided)
    --force         Force re-download even if data exists
    --verify        Verify extraction after download
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# Import kai configuration
try:
    from kai.config.paths import RETRIEVAL_DIR
except ImportError:
    print("Error: kai package not found. Please install kai first:")
    print("  pip install -e .")
    sys.exit(1)


# Zenodo base URL template (will be formatted with record ID and version)
ZENODO_URL_TEMPLATE = "https://zenodo.org/records/{record_id}/files/kai_retrieval_{version}.zip"

# Default Zenodo record ID
DEFAULT_ZENODO_RECORD_ID = "17660667"


class DownloadProgressBar:
    """Simple progress bar for downloads."""

    def __init__(self, total_size: int, description: str = "Downloading"):
        self.total_size = total_size
        self.description = description
        self.downloaded = 0

    def update(self, chunk_size: int):
        """Update progress bar with new chunk."""
        self.downloaded += chunk_size
        if self.total_size > 0:
            percent = (self.downloaded / self.total_size) * 100
            bar_length = 50
            filled = int(bar_length * self.downloaded / self.total_size)
            bar = '=' * filled + '-' * (bar_length - filled)
            size_mb = self.downloaded / (1024 * 1024)
            total_mb = self.total_size / (1024 * 1024)
            print(f'\r{self.description}: [{bar}] {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)',
                  end='', flush=True)

    def finish(self):
        """Complete the progress bar."""
        print()  # New line after progress bar


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 checksum of a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read

    Returns:
        MD5 checksum as hex string
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, destination: Path, force: bool = False) -> bool:
    """Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Path to save the file
        force: Force re-download if file exists

    Returns:
        True if download successful, False otherwise
    """
    if destination.exists() and not force:
        print(f"File already exists: {destination}")
        print("Use --force to re-download")
        return True

    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")

    try:
        # Create request with headers
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'kai-retrieval-downloader/1.0'}
        )

        # Open connection and get file size
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))

            # Create progress bar
            progress = DownloadProgressBar(total_size)

            # Download with progress
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, 'wb') as f:
                while chunk := response.read(8192):
                    f.write(chunk)
                    progress.update(len(chunk))

            progress.finish()

        print(f"✓ Download complete: {destination}")
        return True

    except urllib.error.URLError as e:
        print(f"✗ Download failed: {e}")
        if destination.exists():
            destination.unlink()
        return False
    except Exception as e:
        print(f"✗ Unexpected error during download: {e}")
        if destination.exists():
            destination.unlink()
        return False


def extract_archive(archive_path: Path, destination_dir: Path, verify: bool = False) -> bool:
    """Extract zip archive to destination directory.

    Args:
        archive_path: Path to zip archive
        destination_dir: Directory to extract to
        verify: Verify extraction integrity

    Returns:
        True if extraction successful, False otherwise
    """
    print(f"\nExtracting archive...")
    print(f"Archive: {archive_path}")
    print(f"Destination: {destination_dir}")

    try:
        # Create destination if it doesn't exist
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Extract archive
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"Extracting {total_files} files...")

            # Extract all files with progress
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, destination_dir)
                if i % 100 == 0 or i == total_files:
                    print(f'\rProgress: {i}/{total_files} files ({(i/total_files)*100:.1f}%)',
                          end='', flush=True)

            print()  # New line after progress

        print(f"✓ Extraction complete")

        # Verify extraction if requested
        if verify:
            print("\nVerifying extraction...")
            required_items = ['chromadb', 'collection_registry.json', 'licenses']
            missing = []

            for item in required_items:
                item_path = destination_dir / item
                if not item_path.exists():
                    missing.append(item)

            if missing:
                print(f"✗ Verification failed. Missing items: {', '.join(missing)}")
                return False
            else:
                print("✓ Verification successful")

        return True

    except zipfile.BadZipFile:
        print(f"✗ Error: {archive_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def check_existing_data(retrieval_dir: Path) -> bool:
    """Check if retrieval data already exists.

    Args:
        retrieval_dir: Path to retrieval directory

    Returns:
        True if data exists and appears valid
    """
    required_items = ['chromadb', 'collection_registry.json']

    if not retrieval_dir.exists():
        return False

    for item in required_items:
        if not (retrieval_dir / item).exists():
            return False

    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download and extract kai retrieval database from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download version 251121 (using default Zenodo record)
  python scripts/download_retrieval_data.py 251121

  # Download with custom Zenodo record ID
  python scripts/download_retrieval_data.py 251121 --zenodo-record XXXXX

  # Download from completely custom URL
  python scripts/download_retrieval_data.py 251121 --zenodo-url "https://zenodo.org/records/12345/files/kai_retrieval_251121.zip"

  # Force re-download and verify
  python scripts/download_retrieval_data.py 251121 --force --verify

License Information:
  The retrieval database contains aggregated content from various open-source
  projects. Each project's license is preserved in the 'licenses/' directory.
  The aggregated database is provided under CC-BY-4.0.

  See licenses/README.md in the extracted data for more information.
        """
    )

    parser.add_argument(
        'version',
        type=str,
        help='Version identifier for the retrieval database (e.g., 251121)'
    )

    parser.add_argument(
        '--zenodo-record',
        type=str,
        default=DEFAULT_ZENODO_RECORD_ID,
        help=f'Zenodo record ID (default: {DEFAULT_ZENODO_RECORD_ID})'
    )

    parser.add_argument(
        '--zenodo-url',
        type=str,
        default=None,
        help='Custom Zenodo URL (overrides --zenodo-record and version)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if data exists'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify extraction after download'
    )

    args = parser.parse_args()

    # Determine the download URL
    if args.zenodo_url:
        # Use custom URL if provided
        download_url = args.zenodo_url
    else:
        # Build URL from template using record ID and version
        download_url = ZENODO_URL_TEMPLATE.format(
            record_id=args.zenodo_record,
            version=args.version
        )

    # Determine archive filename
    archive_filename = f"kai_retrieval_{args.version}.zip"

    print("=" * 70)
    print("kai Retrieval Database Download & Setup")
    print("=" * 70)
    print()
    print(f"Version: {args.version}")
    print(f"URL: {download_url}")
    print()

    # Check if data already exists
    if check_existing_data(RETRIEVAL_DIR) and not args.force:
        print(f"✓ Retrieval data already exists at: {RETRIEVAL_DIR}")
        print()
        print("The following items were found:")
        for item in RETRIEVAL_DIR.iterdir():
            print(f"  - {item.name}")
        print()
        print("Use --force to re-download and overwrite existing data")
        return 0

    # Create temporary download directory
    temp_dir = Path.home() / ".kai_agent" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = temp_dir / archive_filename

    # Download archive
    print(f"Target directory: {RETRIEVAL_DIR}")
    print()

    if not download_file(download_url, archive_path, force=args.force):
        print("\n✗ Download failed. Please check:")
        print("  - Your internet connection")
        print("  - The Zenodo URL is correct")
        print("  - Zenodo service is available")
        return 1

    # Extract archive
    if not extract_archive(archive_path, RETRIEVAL_DIR, verify=args.verify):
        print("\n✗ Extraction failed")
        return 1

    # Cleanup
    print("\nCleaning up...")
    archive_path.unlink()
    print(f"✓ Removed temporary file: {archive_path}")

    # Show final status
    print()
    print("=" * 70)
    print("✓ Setup Complete!")
    print("=" * 70)
    print()
    print(f"Retrieval database installed at: {RETRIEVAL_DIR}")
    print()
    print("Contents:")
    for item in sorted(RETRIEVAL_DIR.iterdir()):
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ ({file_count} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")

    print()
    print("You can now use kai with retrieval-augmented generation (RAG)!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
