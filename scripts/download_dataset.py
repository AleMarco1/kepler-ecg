"""
Kepler-ECG: Download dataset

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import argparse    # Handle command-line arguments (e.g., hyperparameter tuning or paths)
import subprocess  # Execute external system commands (e.g., creating venv or installing pip packages)
import sys         # Access system-specific parameters and the Python interpreter path
from pathlib import Path  # Object-oriented management of file system paths (cross-platform compatible)

def check_wget():
    """Verify if wget is installed in the system."""
    try:
        # Run 'wget --version' to check if the executable exists and works
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_dataset(url, output_path):
    print(f"🚀 Starting download from: {url}")
    print(f"📂 Destination: {output_path}")
    
    # Ensure the directory exists (parents=True creates missing intermediate folders)
    output_path.mkdir(parents=True, exist_ok=True)

    # Optimized wget command for PhysioNet:
    # -m: mirror, -np: no-parent (prevents ascending to parent directories)
    # -nH: no-host-directories (avoids creating a folder named 'physionet.org')
    # --cut-dirs: skips the first N directory levels in the URL for a cleaner local path
    # -P: specifies the output directory
    cmd = [
        "wget", "-m", "-np", "-nH",
        "--cut-dirs=3", 
        "-P", str(output_path),
        url
    ]

    try:
        # Run the command and raise an exception if it fails (check=True)
        subprocess.run(cmd, check=True)
        print(f"\n✅ Download completed successfully in: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during download: {e}")
        return False

def main():
    # Initialize the argument parser for a generic PhysioNet downloader
    parser = argparse.ArgumentParser(description="Generic downloader for PhysioNet datasets")
    
    # URL is a required positional argument
    parser.add_argument("url", help="PhysioNet dataset URL (e.g., https://physionet.org/files/ptb-xl/1.0.3/)")
    
    # Default destination folder is set to 'data/raw' for organized project structure
    parser.add_argument("--output", default="data/raw", help="Destination folder (default: data/raw)")

    args = parser.parse_args()

    # Check if the 'wget' tool is available in the system PATH
    if not check_wget():
        print("❌ Error: 'wget' is not installed.")
        print("Installation tips: \n - macOS: 'brew install wget' \n - Linux: 'sudo apt install wget' \n - Windows: Install via Chocolatey or download the binary.")
        sys.exit(1)

    # Convert output string to a Path object and execute the download
    # Path() ensures cross-platform compatibility (Windows vs Linux)
    success = download_dataset(args.url, Path(args.output))
    
    # Exit with status code 0 if successful, 1 otherwise
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()