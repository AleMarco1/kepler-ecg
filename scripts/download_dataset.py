"""
Kepler-ECG: Download Dataset (v2.0 - Multi-Dataset Support)

Downloads ECG datasets from PhysioNet with integrated dataset registry.
Supports: PTB-XL, Chapman, CPSC-2018, Georgia (and more).

Usage:
    # Download by dataset name (recommended):
    python scripts/download_dataset.py --dataset ptb-xl
    python scripts/download_dataset.py --dataset chapman
    python scripts/download_dataset.py --dataset cpsc-2018
    python scripts/download_dataset.py --dataset georgia
    
    # Download all supported datasets:
    python scripts/download_dataset.py --all
    
    # List available datasets:
    python scripts/download_dataset.py --list
    
    # Legacy mode (direct URL):
    python scripts/download_dataset.py https://physionet.org/files/ptb-xl/1.0.3/ --output data/raw/ptb-xl

Author: Alessandro Marconi for Kepler-ECG Project
Version: 2.0.0 - Integrated with DatasetRegistry
Issued on: January 2025
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Try to import from core module
try:
    from core.dataset_registry import get_registry, get_dataset_config
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("⚠️  Warning: core.dataset_registry not found. Using legacy mode only.")


def check_wget() -> bool:
    """Verify if wget is installed in the system."""
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_dataset(url: str, output_path: Path, dataset_name: Optional[str] = None) -> bool:
    """
    Download a dataset from PhysioNet.
    
    Parameters
    ----------
    url : str
        PhysioNet dataset URL
    output_path : Path
        Local destination folder
    dataset_name : str, optional
        Name for display purposes
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    display_name = dataset_name or output_path.name
    
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {display_name}")
    print(f"{'='*60}")
    print(f"🌐 URL: {url}")
    print(f"📂 Destination: {output_path}")
    
    # Ensure the directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate cut-dirs based on URL structure
    # Example: https://physionet.org/files/ptb-xl/1.0.3/ -> 3 dirs to cut
    url_parts = url.rstrip('/').split('/')
    try:
        files_idx = url_parts.index('files')
        cut_dirs = len(url_parts) - files_idx - 1
    except ValueError:
        # For URLs like /content/ instead of /files/
        cut_dirs = 3  # Default
    
    # Optimized wget command for PhysioNet
    cmd = [
        "wget",
        "-m",           # Mirror mode
        "-np",          # No parent (don't ascend to parent directory)
        "-nH",          # No host directories
        f"--cut-dirs={cut_dirs}",  # Skip directory levels
        "-P", str(output_path),
        "--progress=bar:force",  # Show progress bar
        "-e", "robots=off",  # Ignore robots.txt
        url
    ]
    
    try:
        print(f"\n🚀 Starting download...")
        subprocess.run(cmd, check=True)
        print(f"\n✅ Download completed: {display_name}")
        print(f"   Location: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading {display_name}: {e}")
        return False


def list_datasets() -> None:
    """List all available datasets from registry."""
    if not REGISTRY_AVAILABLE:
        print("❌ Dataset registry not available.")
        print("   Make sure src/core/dataset_registry.py exists.")
        return
    
    registry = get_registry()
    
    print("\n" + "="*70)
    print("📋 Available ECG Datasets")
    print("="*70)
    print(f"{'Name':<15} {'Leads':<8} {'Fs (Hz)':<10} {'Description'}")
    print("-"*70)
    
    for name in sorted(registry.available_datasets):
        config = registry.get_config(name)
        print(f"{name:<15} {config.n_leads:<8} {config.sampling_rate:<10} {config.description[:40]}...")
    
    print("-"*70)
    print(f"\nTotal: {len(registry.available_datasets)} datasets")
    print("\nUsage: python scripts/download_dataset.py --dataset <name>")


def download_by_name(dataset_name: str, base_output: Path = Path("data/raw")) -> bool:
    """
    Download a dataset by its registered name.
    
    Parameters
    ----------
    dataset_name : str
        Registered dataset name (e.g., 'ptb-xl', 'chapman')
    base_output : Path
        Base output directory (dataset will be in base_output/dataset_name)
        
    Returns
    -------
    bool
        True if successful
    """
    if not REGISTRY_AVAILABLE:
        print(f"❌ Cannot download '{dataset_name}': registry not available")
        return False
    
    try:
        config = get_dataset_config(dataset_name)
    except ValueError as e:
        print(f"❌ {e}")
        list_datasets()
        return False
    
    if config.physionet_url is None:
        print(f"❌ No PhysioNet URL configured for '{dataset_name}'")
        return False
    
    output_path = base_output / config.name
    return download_dataset(config.physionet_url, output_path, config.name)


def download_all(base_output: Path = Path("data/raw")) -> dict:
    """
    Download all registered datasets.
    
    Returns
    -------
    dict
        Results for each dataset {name: success}
    """
    if not REGISTRY_AVAILABLE:
        print("❌ Cannot download all: registry not available")
        return {}
    
    registry = get_registry()
    results = {}
    
    print("\n" + "="*70)
    print("📦 Downloading ALL datasets")
    print("="*70)
    print(f"Datasets: {', '.join(registry.available_datasets)}")
    print(f"Output: {base_output}")
    
    for name in registry.available_datasets:
        config = registry.get_config(name)
        if config.physionet_url:
            results[name] = download_by_name(name, base_output)
        else:
            print(f"⏭️  Skipping {name}: no PhysioNet URL")
            results[name] = None
    
    # Summary
    print("\n" + "="*70)
    print("📊 Download Summary")
    print("="*70)
    
    for name, success in results.items():
        if success is True:
            print(f"  ✅ {name}")
        elif success is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⏭️  {name} (skipped)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download ECG datasets from PhysioNet (Kepler-ECG v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download PTB-XL:
  python scripts/download_dataset.py --dataset ptb-xl
  
  # Download all datasets:
  python scripts/download_dataset.py --all
  
  # List available datasets:
  python scripts/download_dataset.py --list
  
  # Legacy mode (direct URL):
  python scripts/download_dataset.py https://physionet.org/files/ptb-xl/1.0.3/
        """
    )
    
    # New mode: dataset by name
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset name to download (e.g., ptb-xl, chapman, cpsc-2018, georgia)"
    )
    
    # Download all
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all registered datasets"
    )
    
    # List datasets
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    
    # Legacy mode: direct URL
    parser.add_argument(
        "url",
        nargs="?",
        help="PhysioNet dataset URL (legacy mode)"
    )
    
    # Output directory
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Base output directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    base_output = Path(args.output)
    
    # Check wget first
    if not check_wget():
        print("❌ Error: 'wget' is not installed.")
        print("\nInstallation tips:")
        print("  - macOS:   brew install wget")
        print("  - Linux:   sudo apt install wget")
        print("  - Windows: choco install wget (or download from https://eternallybored.org/misc/wget/)")
        sys.exit(1)
    
    # Handle different modes
    if args.list:
        list_datasets()
        sys.exit(0)
    
    if args.all:
        results = download_all(base_output)
        success_count = sum(1 for v in results.values() if v is True)
        sys.exit(0 if success_count > 0 else 1)
    
    if args.dataset:
        success = download_by_name(args.dataset, base_output)
        sys.exit(0 if success else 1)
    
    if args.url:
        # Legacy mode: direct URL
        # Infer dataset name from URL or use output directory name
        url = args.url
        
        # Try to determine output path
        if args.output == "data/raw":
            # Auto-detect dataset name from URL
            # Example: https://physionet.org/files/ptb-xl/1.0.3/ -> ptb-xl
            parts = url.rstrip('/').split('/')
            for i, part in enumerate(parts):
                if part == 'files' and i + 1 < len(parts):
                    dataset_name = parts[i + 1]
                    output_path = base_output / dataset_name
                    break
            else:
                output_path = base_output / "downloaded"
        else:
            output_path = Path(args.output)
        
        success = download_dataset(url, output_path)
        sys.exit(0 if success else 1)
    
    # No valid arguments provided
    parser.print_help()
    print("\n💡 Tip: Use --list to see available datasets")
    sys.exit(1)


if __name__ == "__main__":
    main()
