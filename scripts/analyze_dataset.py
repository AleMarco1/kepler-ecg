#!/usr/bin/env python3
"""
Kepler-ECG: Dataset Analysis Script (Phase 3)

Esegue l'analisi statistica, visualizzazione, feature selection e preparazione
per Symbolic Regression su qualsiasi dataset ECG processato.

Supporta tutti i dataset del progetto:
- PTB-XL, Chapman, CPSC-2018, Georgia

Usage:
    # Analizza un singolo dataset
    python scripts/analyze_dataset.py --dataset ptb-xl
    
    # Analizza tutti i dataset disponibili
    python scripts/analyze_dataset.py --all
    
    # Solo statistiche (skip visualizzazioni)
    python scripts/analyze_dataset.py --dataset chapman --skip-viz
    
    # Custom input file
    python scripts/analyze_dataset.py --input results/ptb-xl/ptb-xl_features_extracted.csv

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: January 2025
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import analysis modules
try:
    from analysis import (
        StatisticalAnalyzer,
        ECGVisualizer,
        FeatureSelector,
        SymbolicRegressionPrep,
        Phase3Pipeline,
        PipelineConfig,
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import analysis modules: {e}")
    ANALYSIS_AVAILABLE = False

# Import core modules for dataset registry
try:
    from core.dataset_registry import get_dataset_config, get_registry
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

# Superclass categories (matches label_schema.py)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP", "OTHER"]

# Default features to analyze (compressibility-focused)
PRIORITY_FEATURES = [
    "comp_sig_gzip_ratio",
    "comp_sig_lzma_ratio",
    "comp_sig_lempel_ziv_complexity",
    "comp_rr_sample_entropy",
    "comp_rr_approx_entropy",
    "comp_rr_permutation_entropy",
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
    "wav_energy_ratio_low_high",
]


# ============================================================================
# Helper Functions
# ============================================================================

def find_features_file(dataset_name: str) -> Optional[Path]:
    """Find the features file for a dataset."""
    # Try extracted features first
    candidates = [
        Path(f"results/{dataset_name}/{dataset_name}_features_extracted.csv"),
        Path(f"results/{dataset_name}/{dataset_name}_features.csv"),
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


def get_available_datasets() -> List[str]:
    """Get list of datasets with processed features."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    
    available = []
    for d in results_dir.iterdir():
        if d.is_dir():
            if find_features_file(d.name):
                available.append(d.name)
    
    return sorted(available)


def load_features(path: Path) -> pd.DataFrame:
    """Load features from CSV with proper type handling."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns from {path}")
    return df


def get_category_column(df: pd.DataFrame) -> Optional[str]:
    """Find the diagnostic category column."""
    candidates = [
        "diag_primary_category",
        "primary_superclass",
        "diagnosis_primary_category",
    ]
    
    for col in candidates:
        if col in df.columns:
            return col
    
    return None


# ============================================================================
# Analysis Functions
# ============================================================================

def run_statistical_analysis(
    df: pd.DataFrame,
    category_col: str,
    output_dir: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    """Run statistical analysis on features."""
    logger.info("Running statistical analysis...")
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c.startswith(('comp_', 'hrv_', 'wav_', 'morph_', 'interval_'))]
    
    if not feature_cols:
        logger.warning("No feature columns found for statistical analysis")
        return results
    
    # Feature-category association
    try:
        association = analyzer.feature_category_association(
            df, feature_cols, category_col, exclude_categories=["UNKNOWN"]
        )
        
        # Save results
        assoc_path = output_dir / f"{dataset_name}_feature_association.csv"
        association.to_csv(assoc_path, index=False)
        logger.info(f"  Saved: {assoc_path}")
        
        results["feature_association"] = association.to_dict('records')
        results["n_significant_features"] = int((association["is_significant"] == True).sum())
        
        # Top discriminative features
        top_features = association.head(10)["feature"].tolist()
        results["top_features"] = top_features
        logger.info(f"  Top 10 discriminative features: {top_features}")
        
    except Exception as e:
        logger.warning(f"  Feature association failed: {e}")
    
    # Detailed analysis for priority features
    detailed_results = []
    for feature in PRIORITY_FEATURES:
        if feature in df.columns:
            try:
                result = analyzer.compare_categories(
                    df, feature, category_col, exclude_categories=["UNKNOWN"]
                )
                detailed_results.append({
                    "feature": feature,
                    "f_statistic": result.anova.f_statistic,
                    "p_value": result.anova.p_value,
                    "is_significant": result.anova.is_significant,
                    "n_significant_pairs": len(result.get_significant_pairs()),
                })
            except Exception:
                pass
    
    if detailed_results:
        results["priority_features_analysis"] = detailed_results
    
    return results


def run_feature_selection(
    df: pd.DataFrame,
    category_col: str,
    output_dir: Path,
    dataset_name: str,
    n_features: int = 15,
) -> Dict[str, Any]:
    """Run feature selection analysis."""
    logger.info("Running feature selection...")
    
    selector = FeatureSelector()
    results = {}
    
    try:
        # Combined importance
        importance = selector.combined_importance(
            df, category_col, exclude_categories=["UNKNOWN"]
        )
        
        # Save full importance
        imp_path = output_dir / f"{dataset_name}_feature_importance.csv"
        importance.to_csv(imp_path, index=False)
        logger.info(f"  Saved: {imp_path}")
        
        # Get top features
        top_features = importance.head(n_features)["feature"].tolist()
        results["top_features"] = top_features
        results["importance_summary"] = importance.head(n_features).to_dict('records')
        
        # Suggest features for SR
        sr_suggestion = selector.suggest_features_for_sr(
            df, category_col, n_features=12, exclude_categories=["UNKNOWN"]
        )
        results["sr_suggested_features"] = sr_suggestion["suggested_features"]
        
        logger.info(f"  Selected {len(top_features)} top features")
        logger.info(f"  SR suggested: {sr_suggestion['suggested_features'][:5]}...")
        
    except Exception as e:
        logger.warning(f"  Feature selection failed: {e}")
    
    return results


def generate_visualizations(
    df: pd.DataFrame,
    category_col: str,
    output_dir: Path,
    dataset_name: str,
    feature_importance: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Generate visualization plots."""
    logger.info("Generating visualizations...")
    
    viz = ECGVisualizer(dpi=150)
    generated_files = []
    
    # 1. Boxplots for key features
    key_features = [f for f in PRIORITY_FEATURES[:4] if f in df.columns]
    
    for feature in key_features:
        try:
            fig = viz.plot_feature_by_category(
                df, feature, category_col,
                plot_type="boxplot",
                exclude_categories=["UNKNOWN"],
                title=f"{feature} by Diagnosis ({dataset_name})"
            )
            
            filename = f"{dataset_name}_{feature}_boxplot.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(filepath))
            
        except Exception as e:
            logger.warning(f"  Boxplot for {feature} failed: {e}")
    
    # 2. Correlation heatmap for compressibility features
    comp_features = [c for c in df.columns if c.startswith('comp_')]
    if len(comp_features) >= 4:
        try:
            fig = viz.plot_correlation_heatmap(
                df, features=comp_features[:15],
                title=f"Compressibility Feature Correlations ({dataset_name})"
            )
            
            filename = f"{dataset_name}_correlation_heatmap.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(filepath))
            
        except Exception as e:
            logger.warning(f"  Correlation heatmap failed: {e}")
    
    # 3. Feature importance bar chart
    if feature_importance is not None and len(feature_importance) > 0:
        try:
            fig = viz.plot_feature_importance_bar(
                feature_importance, n_features=15,
                title=f"Top Discriminative Features ({dataset_name})"
            )
            
            filename = f"{dataset_name}_feature_importance.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(filepath))
            
        except Exception as e:
            logger.warning(f"  Feature importance plot failed: {e}")
    
    # 4. t-SNE visualization (if enough samples)
    if len(df) >= 500:
        try:
            fig = viz.plot_tsne(
                df, category_col,
                exclude_categories=["UNKNOWN"],
                title=f"t-SNE Projection ({dataset_name})",
                sample_size=2000,
            )
            
            filename = f"{dataset_name}_tsne.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(str(filepath))
            
        except Exception as e:
            logger.warning(f"  t-SNE failed: {e}")
    
    logger.info(f"  Generated {len(generated_files)} visualizations")
    return generated_files


def prepare_sr_datasets(
    df: pd.DataFrame,
    category_col: str,
    output_dir: Path,
    dataset_name: str,
    features: List[str],
) -> Dict[str, Any]:
    """Prepare datasets for Symbolic Regression."""
    logger.info("Preparing SR datasets...")
    
    sr_prep = SymbolicRegressionPrep()
    results = {"datasets": {}, "baseline_performance": {}}
    
    # Filter to valid features
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < 3:
        logger.warning("  Not enough valid features for SR preparation")
        return results
    
    # Get available pathologies in this dataset
    available_cats = df[category_col].dropna().unique().tolist()
    pathologies = [p for p in ["MI", "HYP", "CD", "STTC", "OTHER"] if p in available_cats]
    
    if "NORM" not in available_cats:
        logger.warning("  No NORM category - skipping binary classification datasets")
        return results
    
    sr_output_dir = output_dir / "sr_ready"
    sr_output_dir.mkdir(exist_ok=True)
    
    for pathology in pathologies:
        try:
            # Check if enough samples
            n_pos = len(df[df[category_col] == pathology])
            n_neg = len(df[df[category_col] == "NORM"])
            
            if n_pos < 50 or n_neg < 50:
                logger.info(f"  Skipping {pathology} (insufficient samples: {n_pos} vs {n_neg})")
                continue
            
            dataset = sr_prep.prepare_binary_classification(
                df,
                positive_class=pathology,
                negative_class="NORM",
                category_col=category_col,
                features=valid_features,
                normalize=True,
                balance_classes=True,
            )
            
            # Save dataset
            filename = f"{dataset_name}_norm_vs_{pathology.lower()}.csv"
            filepath = sr_output_dir / filename
            dataset.save(str(filepath))
            
            results["datasets"][pathology] = {
                "path": str(filepath),
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "positive_count": dataset.metadata["positive_count"],
                "negative_count": dataset.metadata["negative_count"],
            }
            
            # Baseline performance
            try:
                baseline = sr_prep.get_baseline_performance(dataset)
                results["baseline_performance"][pathology] = baseline
            except Exception:
                pass
            
            logger.info(f"  Created: {filename} ({dataset.n_samples} samples)")
            
        except Exception as e:
            logger.warning(f"  Failed for {pathology}: {e}")
    
    return results


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_dataset(
    dataset_name: str,
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_viz: bool = False,
    n_features: int = 15,
) -> Dict[str, Any]:
    """
    Run complete Phase 3 analysis on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        input_path: Path to features CSV (auto-detected if None)
        output_dir: Output directory (auto-created if None)
        skip_viz: Skip visualization generation
        n_features: Number of top features to select
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info("="*60)
    logger.info(f"KEPLER-ECG PHASE 3 ANALYSIS: {dataset_name.upper()}")
    logger.info("="*60)
    
    # Find input file
    if input_path is None:
        input_path = find_features_file(dataset_name)
        if input_path is None:
            raise FileNotFoundError(
                f"No features file found for {dataset_name}. "
                f"Run extract_features_dataset.py first."
            )
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"results/{dataset_name}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"\n[1/5] Loading data from: {input_path}")
    df = load_features(input_path)
    
    # Find category column
    category_col = get_category_column(df)
    if category_col is None:
        logger.warning("No diagnostic category column found - limited analysis available")
        category_col = "primary_superclass"  # Try anyway
    else:
        logger.info(f"  Using category column: {category_col}")
        
        # Show distribution
        if category_col in df.columns:
            dist = df[category_col].value_counts()
            logger.info(f"  Category distribution:")
            for cat, count in dist.items():
                logger.info(f"    {cat}: {count} ({100*count/len(df):.1f}%)")
    
    # Initialize results
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "n_records": len(df),
        "n_columns": len(df.columns),
        "category_column": category_col,
    }
    
    # Import matplotlib only if needed
    if not skip_viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    # Run statistical analysis
    logger.info(f"\n[2/5] Statistical Analysis")
    stat_results = run_statistical_analysis(df, category_col, output_dir, dataset_name)
    results["statistical"] = stat_results
    
    # Run feature selection
    logger.info(f"\n[3/5] Feature Selection")
    selection_results = run_feature_selection(df, category_col, output_dir, dataset_name, n_features)
    results["feature_selection"] = selection_results
    
    # Load feature importance for visualization
    feature_importance = None
    imp_path = output_dir / f"{dataset_name}_feature_importance.csv"
    if imp_path.exists():
        feature_importance = pd.read_csv(imp_path)
    
    # Generate visualizations
    if not skip_viz:
        logger.info(f"\n[4/5] Generating Visualizations")
        viz_files = generate_visualizations(
            df, category_col, output_dir, dataset_name, feature_importance
        )
        results["visualizations"] = viz_files
    else:
        logger.info(f"\n[4/5] Skipping Visualizations")
        results["visualizations"] = []
    
    # Prepare SR datasets
    logger.info(f"\n[5/5] Preparing SR Datasets")
    sr_features = selection_results.get("sr_suggested_features", PRIORITY_FEATURES)
    sr_results = prepare_sr_datasets(df, category_col, output_dir, dataset_name, sr_features)
    results["sr_preparation"] = sr_results
    
    # Save summary
    summary_path = output_dir / f"{dataset_name}_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved summary: {summary_path}")
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Records: {results['n_records']:,}")
    logger.info(f"Significant features: {stat_results.get('n_significant_features', 'N/A')}")
    logger.info(f"SR datasets created: {len(sr_results.get('datasets', {}))}")
    logger.info(f"Visualizations: {len(results['visualizations'])}")
    logger.info(f"Output directory: {output_dir}")
    
    return results


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Phase 3 Analysis (Statistical, Visualization, SR Prep)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single dataset
  python scripts/analyze_dataset.py --dataset ptb-xl
  
  # Analyze all available datasets
  python scripts/analyze_dataset.py --all
  
  # Skip visualizations (faster)
  python scripts/analyze_dataset.py --dataset chapman --skip-viz
  
  # Custom input file
  python scripts/analyze_dataset.py --input results/custom/features.csv --name custom
  
  # Specify number of top features
  python scripts/analyze_dataset.py --dataset ptb-xl --n-features 20
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name (e.g., ptb-xl, chapman, cpsc-2018, georgia)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Analyze all available datasets')
    parser.add_argument('--input', '-i', type=str,
                        help='Custom input features CSV path')
    parser.add_argument('--name', type=str,
                        help='Dataset name when using --input')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (default: results/{dataset}/analysis)')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--n-features', '-n', type=int, default=15,
                        help='Number of top features to select (default: 15)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available datasets and exit')
    
    args = parser.parse_args()
    
    # List available datasets
    if args.list:
        available = get_available_datasets()
        if available:
            print("Available datasets with features:")
            for ds in available:
                path = find_features_file(ds)
                print(f"  - {ds}: {path}")
        else:
            print("No datasets with features found in results/")
        return 0
    
    # Validate arguments
    if not args.dataset and not args.all and not args.input:
        parser.error("Specify --dataset, --all, or --input")
    
    if args.input and not args.name:
        parser.error("--name is required when using --input")
    
    # Check analysis modules
    if not ANALYSIS_AVAILABLE:
        print("❌ Error: Analysis modules not available")
        print("   Make sure src/analysis/ is properly set up")
        return 1
    
    # Determine datasets to process
    if args.all:
        datasets = get_available_datasets()
        if not datasets:
            print("❌ No datasets with features found")
            return 1
        print(f"Found {len(datasets)} datasets to analyze: {datasets}")
    elif args.input:
        datasets = [(args.name, Path(args.input))]
    else:
        datasets = [args.dataset]
    
    # Process each dataset
    all_results = {}
    
    for dataset in datasets:
        if isinstance(dataset, tuple):
            name, path = dataset
        else:
            name = dataset
            path = None
        
        try:
            output_dir = Path(args.output) if args.output else None
            
            results = analyze_dataset(
                dataset_name=name,
                input_path=path,
                output_dir=output_dir,
                skip_viz=args.skip_viz,
                n_features=args.n_features,
            )
            
            all_results[name] = results
            
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary for multi-dataset
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("MULTI-DATASET SUMMARY")
        print("="*60)
        for name, res in all_results.items():
            n_sr = len(res.get("sr_preparation", {}).get("datasets", {}))
            print(f"  {name}: {res['n_records']:,} records, {n_sr} SR datasets")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
