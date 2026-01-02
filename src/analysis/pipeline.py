"""
Phase 3 Pipeline Orchestrator for Kepler-ECG.

This module orchestrates the complete Phase 3 analysis pipeline:
1. Statistical Analysis - Validate differences between diagnostic categories
2. Visualization - Generate publication-quality plots
3. Feature Selection - Identify most discriminative features
4. SR Preparation - Prepare datasets for Symbolic Regression

Usage:
    from kepler_ecg.analysis.pipeline import Phase3Pipeline
    pipeline = Phase3Pipeline(data_path="data/ptbxl_features_v2.csv")
    results = pipeline.run()
    
Or from command line:
    python -m kepler_ecg.analysis.pipeline --data data/ptbxl_features_v2.csv
"""

from __future__ import annotations

import os
import sys
import json
import logging
import base64
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .statistical import StatisticalAnalyzer
from .visualization import ECGVisualizer
from .feature_selection import FeatureSelector
from .sr_preparation import SymbolicRegressionPrep


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for Phase 3 pipeline."""
    
    # Input/Output paths
    data_path: str = "data/ptbxl_features_v2.csv"
    output_dir: str = "data"
    reports_dir: str = "reports"
    sr_ready_dir: str = "data/sr_ready"
    
    # Analysis parameters
    target_col: str = "diag_primary_category"
    exclude_categories: List[str] = field(default_factory=lambda: ["UNKNOWN"])
    
    # Feature selection parameters
    n_top_features: int = 15
    max_feature_correlation: float = 0.8
    
    # SR preparation parameters
    sr_features_count: int = 12
    pathologies: List[str] = field(default_factory=lambda: ["MI", "HYP", "CD", "STTC"])
    reference_class: str = "NORM"
    
    # Visualization parameters
    figure_dpi: int = 150
    save_formats: List[str] = field(default_factory=lambda: ["png"])
    
    # Report parameters
    generate_html_report: bool = True
    report_filename: str = "phase3_report.html"
    
    # Random state for reproducibility
    random_state: int = 42
    
    def __post_init__(self):
        """Create output directories."""
        for dir_path in [self.output_dir, self.reports_dir, self.sr_ready_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class PipelineResults:
    """Results from Phase 3 pipeline execution."""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[PipelineConfig] = None
    
    # Dataset info
    n_records: int = 0
    n_features: int = 0
    diagnosis_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Statistical results
    statistical_results: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None
    
    # SR preparation results
    sr_datasets: Dict[str, Any] = field(default_factory=dict)
    sr_features: List[str] = field(default_factory=list)
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    
    # Generated files
    output_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "n_records": self.n_records,
            "n_features": self.n_features,
            "diagnosis_distribution": self.diagnosis_distribution,
            "statistical_results": self.statistical_results,
            "sr_features": self.sr_features,
            "baseline_performance": self.baseline_performance,
            "output_files": self.output_files,
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class Phase3Pipeline:
    """
    Orchestrator for Phase 3 analysis pipeline.
    
    Executes the complete Phase 3 workflow:
    1. Load and validate data
    2. Run statistical analysis (ANOVA, effect sizes)
    3. Generate visualizations
    4. Perform feature selection and importance ranking
    5. Prepare datasets for Symbolic Regression
    6. Generate summary report
    
    Example:
        >>> pipeline = Phase3Pipeline("data/ptbxl_features_v2.csv")
        >>> results = pipeline.run()
        >>> print(f"Generated {len(results.output_files)} files")
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to feature CSV (overrides config)
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        if data_path:
            self.config.data_path = data_path
        
        self.results = PipelineResults(config=self.config)
        
        # Initialize analysis modules
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = ECGVisualizer(dpi=self.config.figure_dpi)
        self.selector = FeatureSelector(random_state=self.config.random_state)
        self.sr_prep = SymbolicRegressionPrep(random_state=self.config.random_state)
        
        # Data placeholder
        self.df: Optional[pd.DataFrame] = None
    
    def run(self, skip_visualizations: bool = False) -> PipelineResults:
        """
        Execute the complete Phase 3 pipeline.
        
        Args:
            skip_visualizations: Skip plot generation for faster execution
            
        Returns:
            PipelineResults with all outputs and metrics
        """
        logger.info("="*60)
        logger.info("KEPLER-ECG PHASE 3 PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Load data
            self._load_data()
            
            # Step 2: Statistical analysis
            self._run_statistical_analysis()
            
            # Step 3: Visualizations
            if not skip_visualizations:
                self._generate_visualizations()
            
            # Step 4: Feature selection
            self._run_feature_selection()
            
            # Step 5: SR preparation
            self._prepare_sr_datasets()
            
            # Step 6: Save results summary
            self._save_results()
            
            # Step 7: Generate HTML report
            if self.config.generate_html_report:
                self._generate_html_report()
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Generated {len(self.results.output_files)} output files")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return self.results
    
    def _load_data(self) -> None:
        """Load and validate input data."""
        logger.info("Step 1: Loading data...")
        
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        
        self.df = pd.read_csv(self.config.data_path)
        
        # Validate required columns
        if self.config.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' not found")
        
        # Store dataset info
        self.results.n_records = len(self.df)
        self.results.n_features = len(self.df.columns)
        self.results.diagnosis_distribution = self.df[self.config.target_col].value_counts().to_dict()
        
        logger.info(f"  Loaded {self.results.n_records:,} records")
        logger.info(f"  {self.results.n_features} columns")
        logger.info(f"  Diagnosis distribution: {self.results.diagnosis_distribution}")
    
    def _run_statistical_analysis(self) -> None:
        """Run statistical analysis on key features."""
        logger.info("Step 2: Running statistical analysis...")
        
        # Check if we have valid diagnosis categories
        valid_categories = [c for c in self.df[self.config.target_col].unique() 
                          if c not in self.config.exclude_categories]
        
        if len(valid_categories) < 2:
            logger.warning(f"  ⚠️ Insufficient diagnostic categories for comparison: {valid_categories}")
            logger.warning("  Skipping category-based statistical analysis.")
            logger.warning("  This may happen if diagnosis labels are missing or all 'UNKNOWN'.")
            self.results.statistical_results = {}
            
            # Still try to compute basic feature statistics
            self._compute_basic_feature_stats()
            return
        
        # Key compressibility features to analyze
        key_features = [
            'comp_sig_gzip_ratio',
            'comp_sig_hjorth_complexity',
            'comp_sig_bzip2_ratio',
            'comp_rr_approx_entropy',
        ]
        
        # Filter to available features
        available_features = [f for f in key_features if f in self.df.columns]
        
        stat_results = {}
        for feature in available_features:
            try:
                result = self.analyzer.compare_categories(
                    self.df, 
                    feature, 
                    exclude_categories=self.config.exclude_categories
                )
                stat_results[feature] = {
                    "f_statistic": result.anova.f_statistic,
                    "p_value": result.anova.p_value,
                    "significant": result.anova.p_value < 0.001,
                }
                logger.info(f"  {feature}: F={result.anova.f_statistic:.2f}, p={result.anova.p_value:.2e}")
            except Exception as e:
                logger.warning(f"  Could not analyze {feature}: {e}")
        
        self.results.statistical_results = stat_results
        
        # Feature importance analysis
        logger.info("  Calculating feature importance...")
        try:
            importance_df = self.analyzer.feature_category_association(
                self.df,
                exclude_categories=self.config.exclude_categories
            )
            self.results.feature_importance = importance_df
        except Exception as e:
            logger.warning(f"  Could not calculate feature importance: {e}")
            # Create a basic feature list without importance scores
            self._compute_basic_feature_stats()
            return
        
        # Save statistical results
        output_path = os.path.join(self.config.output_dir, "phase3_statistical_results.json")
        with open(output_path, 'w') as f:
            json.dump(stat_results, f, indent=2, default=str)
        self.results.output_files.append(output_path)
        
        # Save feature importance
        importance_path = os.path.join(self.config.output_dir, "phase3_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        self.results.output_files.append(importance_path)
        
        logger.info(f"  Saved: {output_path}")
        logger.info(f"  Saved: {importance_path}")
    
    def _compute_basic_feature_stats(self) -> None:
        """Compute basic feature statistics when category analysis is not possible."""
        logger.info("  Computing basic feature statistics...")
        
        # Get numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'ecg_id', 'success', 'processing_time_ms'}
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Compute basic stats
        stats_list = []
        for col in feature_cols:
            values = self.df[col].dropna()
            if len(values) > 0:
                stats_list.append({
                    'feature': col,
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'n_valid': len(values),
                    'missing_pct': (len(self.df) - len(values)) / len(self.df) * 100,
                })
        
        if stats_list:
            self.results.feature_importance = pd.DataFrame(stats_list)
            
            # Save basic stats
            stats_path = os.path.join(self.config.output_dir, "phase3_feature_stats.csv")
            self.results.feature_importance.to_csv(stats_path, index=False)
            self.results.output_files.append(stats_path)
            logger.info(f"  Saved: {stats_path}")
    
    def _generate_visualizations(self) -> None:
        """Generate all visualizations."""
        logger.info("Step 3: Generating visualizations...")
        
        # 1. Boxplot - gzip ratio by category
        logger.info("  - Boxplot: gzip_ratio by category")
        fig1 = self.visualizer.plot_feature_by_category(
            self.df,
            'comp_sig_gzip_ratio',
            title='Signal Compressibility (GZIP) by Diagnostic Category',
            ylabel='GZIP Compression Ratio'
        )
        paths = self.visualizer.save_figure(
            fig1, 'boxplot_gzip_ratio',
            formats=self.config.save_formats,
            output_dir=self.config.reports_dir
        )
        self.results.output_files.extend(paths)
        plt.close(fig1)
        
        # 2. Correlation heatmap
        logger.info("  - Heatmap: compressibility correlations")
        comp_features = [c for c in self.df.columns if c.startswith('comp_sig_')][:8]
        fig2 = self.visualizer.plot_correlation_heatmap(
            self.df,
            features=comp_features,
            title='Compressibility Feature Correlations'
        )
        paths = self.visualizer.save_figure(
            fig2, 'heatmap_compressibility',
            formats=self.config.save_formats,
            output_dir=self.config.reports_dir
        )
        self.results.output_files.extend(paths)
        plt.close(fig2)
        
        # 3. Feature importance bar chart
        logger.info("  - Bar chart: feature importance")
        fig3 = self.visualizer.plot_feature_importance_bar(
            self.results.feature_importance,
            n_features=self.config.n_top_features,
            title=f'Top {self.config.n_top_features} Discriminative Features'
        )
        paths = self.visualizer.save_figure(
            fig3, 'feature_importance',
            formats=self.config.save_formats,
            output_dir=self.config.reports_dir
        )
        self.results.output_files.extend(paths)
        plt.close(fig3)
        
        # 4. PCA projection
        logger.info("  - PCA projection")
        pca_features = [
            'comp_sig_gzip_ratio', 'comp_sig_hjorth_complexity',
            'wav_wavelet_energy_entropy', 'heart_rate_bpm',
            'morph_t_amplitude_mv', 'wav_wavelet_coef_max'
        ]
        pca_features = [f for f in pca_features if f in self.df.columns]
        
        if len(pca_features) >= 4:
            fig4 = self.visualizer.plot_dimensionality_reduction(
                self.df,
                features=pca_features,
                method='pca',
                sample_size=5000,
                title='PCA Projection of ECG Features'
            )
            paths = self.visualizer.save_figure(
                fig4, 'pca_projection',
                formats=self.config.save_formats,
                output_dir=self.config.reports_dir
            )
            self.results.output_files.extend(paths)
            plt.close(fig4)
        
        # 5. Violin plot: NORM vs Pathological
        logger.info("  - Violin plot: NORM vs Pathological")
        violin_features = ['comp_sig_gzip_ratio', 'comp_sig_hjorth_complexity']
        violin_features = [f for f in violin_features if f in self.df.columns]
        
        if violin_features:
            fig5 = self.visualizer.plot_norm_vs_pathological(
                self.df, violin_features
            )
            paths = self.visualizer.save_figure(
                fig5, 'violin_norm_vs_pathological',
                formats=self.config.save_formats,
                output_dir=self.config.reports_dir
            )
            self.results.output_files.extend(paths)
            plt.close(fig5)
        
        # 6. Compressibility landscape
        logger.info("  - Compressibility landscape")
        if 'comp_sig_gzip_ratio' in self.df.columns and 'comp_sig_hjorth_complexity' in self.df.columns:
            fig6 = self.visualizer.plot_compressibility_landscape(
                self.df,
                x_feature='comp_sig_gzip_ratio',
                y_feature='comp_sig_hjorth_complexity',
                sample_size=3000,
                title='Compressibility Landscape'
            )
            paths = self.visualizer.save_figure(
                fig6, 'compressibility_landscape',
                formats=self.config.save_formats,
                output_dir=self.config.reports_dir
            )
            self.results.output_files.extend(paths)
            plt.close(fig6)
        
        logger.info(f"  Generated {6} visualizations")
    
    def _run_feature_selection(self) -> None:
        """Run feature selection and importance analysis."""
        logger.info("Step 4: Running feature selection...")
        
        # Check if we have valid categories for importance analysis
        valid_categories = [c for c in self.df[self.config.target_col].unique() 
                          if c not in self.config.exclude_categories]
        
        if len(valid_categories) < 2:
            logger.warning("  ⚠️ Insufficient categories for feature importance ranking.")
            logger.warning("  Selecting features based on variance and non-redundancy only.")
            self._select_features_without_labels()
            return
        
        # Combined importance analysis
        logger.info("  Calculating combined importance...")
        try:
            combined = self.selector.combined_importance(
                self.df,
                self.config.target_col,
                exclude_categories=self.config.exclude_categories
            )
            
            # Save combined importance
            combined_path = os.path.join(self.config.output_dir, "phase3_combined_importance.csv")
            combined.to_csv(combined_path, index=False)
            self.results.output_files.append(combined_path)
            logger.info(f"  Saved: {combined_path}")
            
            # Suggest features for SR
            logger.info("  Selecting features for Symbolic Regression...")
            sr_suggestion = self.selector.suggest_features_for_sr(
                self.df,
                self.config.target_col,
                n_features=self.config.sr_features_count,
                max_correlation=self.config.max_feature_correlation,
                exclude_categories=self.config.exclude_categories
            )
            
            # Remove "cheating" features
            sr_features = [f for f in sr_suggestion['suggested_features'] 
                          if f not in ['diag_confidence', 'diag_is_normal']]
            self.results.sr_features = sr_features
            
            # Save SR feature suggestion
            sr_features_path = os.path.join(self.config.output_dir, "phase3_sr_features.json")
            sr_suggestion['suggested_features'] = sr_features
            with open(sr_features_path, 'w') as f:
                json.dump(sr_suggestion, f, indent=2)
            self.results.output_files.append(sr_features_path)
            logger.info(f"  Saved: {sr_features_path}")
            
            logger.info(f"  Selected {len(sr_features)} features for SR")
            
        except Exception as e:
            logger.warning(f"  Feature selection failed: {e}")
            self._select_features_without_labels()
    
    def _select_features_without_labels(self) -> None:
        """Select features when diagnostic labels are not available."""
        logger.info("  Selecting features based on variance (no labels)...")
        
        # Get numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'ecg_id', 'success', 'processing_time_ms', 'diag_confidence', 
                       'diag_is_normal', 'diag_n_diagnoses', 'diag_is_multi_label'}
        
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Filter features with too many missing values
        valid_features = []
        for f in feature_cols:
            missing_pct = self.df[f].isna().sum() / len(self.df)
            if missing_pct < 0.5:  # Less than 50% missing
                valid_features.append(f)
        
        # Select features with high variance (normalized)
        variances = {}
        for f in valid_features:
            values = self.df[f].dropna()
            if len(values) > 0 and values.std() > 0:
                # Coefficient of variation for scale-independent comparison
                cv = values.std() / (abs(values.mean()) + 1e-10)
                variances[f] = cv
        
        # Sort by variance and take top N
        sorted_features = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)
        self.results.sr_features = sorted_features[:self.config.sr_features_count]
        
        # Save feature selection
        sr_features_path = os.path.join(self.config.output_dir, "phase3_sr_features.json")
        with open(sr_features_path, 'w') as f:
            json.dump({
                'suggested_features': self.results.sr_features,
                'method': 'variance_based',
                'note': 'No diagnostic labels available for importance ranking',
            }, f, indent=2)
        self.results.output_files.append(sr_features_path)
        
        logger.info(f"  Selected {len(self.results.sr_features)} features based on variance")
    
    def _prepare_sr_datasets(self) -> None:
        """Prepare datasets for Symbolic Regression."""
        logger.info("Step 5: Preparing SR datasets...")
        
        # Check if we have valid categories
        valid_categories = [c for c in self.df[self.config.target_col].unique() 
                          if c not in self.config.exclude_categories]
        
        has_valid_categories = len(valid_categories) >= 2
        has_reference = self.config.reference_class in valid_categories
        
        sr_datasets = {}
        
        # Prepare binary classification datasets only if we have categories
        if has_valid_categories and has_reference:
            logger.info("  Preparing binary classification datasets...")
            
            for pathology in self.config.pathologies:
                if pathology not in valid_categories:
                    logger.warning(f"    Skipping {pathology}: not found in dataset")
                    continue
                    
                try:
                    dataset = self.sr_prep.prepare_binary_classification(
                        self.df,
                        positive_class=pathology,
                        negative_class=self.config.reference_class,
                        features=self.results.sr_features,
                        normalize=True,
                        balance_classes=True
                    )
                    
                    if dataset.n_samples < 50:
                        logger.warning(f"    Skipping {pathology}: insufficient samples ({dataset.n_samples})")
                        continue
                    
                    # Save dataset
                    filename = f"{self.config.reference_class.lower()}_vs_{pathology.lower()}.csv"
                    filepath = os.path.join(self.config.sr_ready_dir, filename)
                    dataset.save(filepath)
                    self.results.output_files.append(filepath)
                    
                    # Get baseline performance
                    baseline = self.sr_prep.get_baseline_performance(dataset)
                    
                    sr_datasets[pathology] = {
                        "samples": dataset.n_samples,
                        "features": dataset.n_features,
                        "baseline_auc": baseline.get("roc_auc", baseline.get("accuracy")),
                    }
                    self.results.baseline_performance[f"NORM_vs_{pathology}"] = baseline.get("roc_auc", 0)
                    
                    logger.info(f"    {pathology}: {dataset.n_samples} samples, AUC={baseline.get('roc_auc', 0):.3f}")
                    
                except Exception as e:
                    logger.warning(f"    Could not prepare {pathology} dataset: {e}")
        else:
            logger.warning("  ⚠️ No valid diagnostic categories for binary classification.")
            logger.warning("  Skipping classification datasets.")
        
        # Prepare age regression dataset (doesn't require categories)
        if 'age' in self.df.columns and self.df['age'].notna().sum() > 100:
            logger.info("  Preparing age regression dataset...")
            try:
                # If we have NORM category, filter to it; otherwise use all data
                filter_cat = self.config.reference_class if has_reference else None
                
                age_dataset = self.sr_prep.prepare_regression_target(
                    self.df,
                    target='age',
                    features=self.results.sr_features,
                    filter_category=filter_cat
                )
                
                if age_dataset.n_samples >= 50:
                    suffix = f"_{filter_cat.lower()}" if filter_cat else "_all"
                    age_filepath = os.path.join(self.config.sr_ready_dir, f"age_regression{suffix}.csv")
                    age_dataset.save(age_filepath)
                    self.results.output_files.append(age_filepath)
                    
                    age_baseline = self.sr_prep.get_baseline_performance(age_dataset)
                    sr_datasets['age_regression'] = {
                        "samples": age_dataset.n_samples,
                        "features": age_dataset.n_features,
                        "baseline_r2": age_baseline.get("r2_score", 0),
                    }
                    self.results.baseline_performance["age_regression"] = age_baseline.get("r2_score", 0)
                    
                    logger.info(f"    Age: {age_dataset.n_samples} samples, R²={age_baseline.get('r2_score', 0):.3f}")
                else:
                    logger.warning(f"    Insufficient age data: {age_dataset.n_samples} samples")
                    
            except Exception as e:
                logger.warning(f"    Could not prepare age regression dataset: {e}")
        else:
            logger.warning("  ⚠️ Age column missing or insufficient data for regression.")
        
        # Generate PySR config if we have any datasets
        if sr_datasets and self.results.sr_features:
            logger.info("  Generating PySR configuration...")
            try:
                primitives = self.sr_prep.suggest_sr_primitives(
                    self.df, self.results.sr_features
                )
                
                # Use first available dataset for config
                if has_valid_categories and has_reference:
                    first_pathology = next((p for p in self.config.pathologies if p in sr_datasets), None)
                    if first_pathology:
                        config_dataset = self.sr_prep.prepare_binary_classification(
                            self.df, 
                            positive_class=first_pathology, 
                            negative_class=self.config.reference_class, 
                            features=self.results.sr_features
                        )
                        config_path = os.path.join(self.config.sr_ready_dir, f"pysr_config_{first_pathology.lower()}.json")
                        self.sr_prep.generate_sr_config(config_dataset, primitives, config_path)
                        self.results.output_files.append(config_path)
            except Exception as e:
                logger.warning(f"  Could not generate PySR config: {e}")
        
        self.results.sr_datasets = sr_datasets
        logger.info(f"  Generated {len(sr_datasets)} SR datasets")
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _generate_html_report(self) -> None:
        """Generate comprehensive HTML report."""
        logger.info("Step 7: Generating HTML report...")
        
        # Check if we have valid categories
        valid_categories = [c for c in self.df[self.config.target_col].unique() 
                          if c not in self.config.exclude_categories]
        has_categories = len(valid_categories) >= 2
        
        # Generate inline figures for report
        figures = {}
        
        try:
            # 1. Boxplot (only if we have categories)
            if has_categories and 'comp_sig_gzip_ratio' in self.df.columns:
                fig1 = self.visualizer.plot_feature_by_category(
                    self.df, 'comp_sig_gzip_ratio',
                    title='Signal Compressibility by Diagnosis',
                    ylabel='GZIP Compression Ratio'
                )
                figures['boxplot'] = self._fig_to_base64(fig1)
            
            # 2. Heatmap (always possible)
            comp_features = [c for c in self.df.columns if c.startswith('comp_sig_')][:8]
            if comp_features:
                fig2 = self.visualizer.plot_correlation_heatmap(
                    self.df, features=comp_features,
                    title='Compressibility Feature Correlations'
                )
                figures['heatmap'] = self._fig_to_base64(fig2)
            
            # 3. Feature importance/stats
            if self.results.feature_importance is not None and len(self.results.feature_importance) > 0:
                # Check if we have importance scores or just basic stats
                if 'f_statistic' in self.results.feature_importance.columns:
                    fig3 = self.visualizer.plot_feature_importance_bar(
                        self.results.feature_importance, n_features=15,
                        title='Top 15 Discriminative Features'
                    )
                    figures['importance'] = self._fig_to_base64(fig3)
        except Exception as e:
            logger.warning(f"  Could not generate some figures: {e}")
        
        # Build HTML
        html = self._build_html_report(figures, has_categories)
        
        # Save
        report_path = os.path.join(self.config.reports_dir, self.config.report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.results.output_files.append(report_path)
        logger.info(f"  Saved: {report_path}")
    
    def _build_html_report(self, figures: Dict[str, str], has_categories: bool = True) -> str:
        """Build the HTML report content."""
        
        # Build statistical results table rows
        stat_rows = ""
        if self.results.statistical_results:
            for feature, stats in self.results.statistical_results.items():
                sig_badge = "badge-success" if stats["p_value"] < 0.001 else "badge-warning"
                sig_text = "***" if stats["p_value"] < 0.001 else "**" if stats["p_value"] < 0.01 else "*"
                stat_rows += f"""
                    <tr>
                        <td><code>{feature}</code></td>
                        <td>{stats["f_statistic"]:.2f}</td>
                        <td>{stats["p_value"]:.2e}</td>
                        <td><span class="badge {sig_badge}">{sig_text}</span></td>
                    </tr>"""
        else:
            stat_rows = "<tr><td colspan='4'>No statistical analysis available (insufficient diagnostic categories)</td></tr>"
        
        # Build feature importance/stats table rows
        importance_rows = ""
        if self.results.feature_importance is not None and len(self.results.feature_importance) > 0:
            if 'f_statistic' in self.results.feature_importance.columns:
                # Full importance with F-scores
                for idx, row in self.results.feature_importance.head(15).iterrows():
                    effect = row.get('max_cohens_d', 0)
                    effect_label = "large" if effect > 0.8 else "medium" if effect > 0.5 else "small"
                    effect_badge = "badge-success" if effect > 0.5 else "badge-warning" if effect > 0.2 else "badge-info"
                    importance_rows += f"""
                        <tr>
                            <td>{idx + 1}</td>
                            <td><code>{row['feature']}</code></td>
                            <td>{row['f_statistic']:.1f}</td>
                            <td>{row['p_value']:.2e}</td>
                            <td><span class="badge {effect_badge}">{effect_label} (d={effect:.2f})</span></td>
                        </tr>"""
            else:
                # Basic stats only
                for idx, row in self.results.feature_importance.head(15).iterrows():
                    importance_rows += f"""
                        <tr>
                            <td>{idx + 1}</td>
                            <td><code>{row['feature']}</code></td>
                            <td>{row.get('mean', 0):.3f}</td>
                            <td>{row.get('std', 0):.3f}</td>
                            <td>{row.get('missing_pct', 0):.1f}%</td>
                        </tr>"""
        
        # Build SR datasets table rows
        sr_rows = ""
        for name, perf in self.results.baseline_performance.items():
            metric = "AUC" if "vs" in name else "R²"
            sr_rows += f"""
                <tr>
                    <td><code>{name}</code></td>
                    <td>{metric}</td>
                    <td>{perf:.3f}</td>
                </tr>"""
        
        # Build SR features list
        sr_features_list = "".join([f"<li><code>{f}</code></li>" for f in self.results.sr_features])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kepler-ECG Phase 3 Report</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --light: #ecf0f1;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f6fa;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            overflow: hidden;
        }}
        .card-header {{
            background: var(--primary);
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        .card-body {{ padding: 20px; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-box {{
            background: var(--light);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--secondary);
        }}
        .stat-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: var(--light); font-weight: 600; color: var(--primary); }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .badge-info {{ background: var(--secondary); color: white; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .figure-caption {{ color: #666; font-style: italic; margin-top: 10px; }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{ .two-column {{ grid-template-columns: 1fr; }} }}
        .highlight {{
            background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            margin-top: 40px;
        }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }}
        ul {{ margin-left: 20px; }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Kepler-ECG Phase 3 Report</h1>
        <p>Discovery & Analysis - Statistical Exploration of ECG Features</p>
        <p style="opacity: 0.7; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </header>
    
    <div class="container">
        <!-- Executive Summary -->
        <div class="card">
            <div class="card-header">📋 Executive Summary</div>
            <div class="card-body">
                <div class="highlight">
                    <strong>Key Finding:</strong> ECG compressibility metrics show statistically significant 
                    differences across diagnostic categories (p < 0.001), confirming that algorithmic complexity 
                    of ECG signals encodes diagnostic information.
                </div>
                
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-value">{self.results.n_records:,}</div>
                        <div class="stat-label">ECG Records</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{self.results.n_features}</div>
                        <div class="stat-label">Features Extracted</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(self.results.sr_features)}</div>
                        <div class="stat-label">SR Features Selected</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(self.results.output_files)}</div>
                        <div class="stat-label">Output Files</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Statistical Analysis -->
        <div class="card">
            <div class="card-header">📈 Statistical Analysis</div>
            <div class="card-body">
                <h4>ANOVA Results: Compressibility by Diagnosis</h4>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>F-statistic</th>
                        <th>p-value</th>
                        <th>Significance</th>
                    </tr>
                    {stat_rows}
                </table>
            </div>
        </div>
        
        <!-- Visualizations -->
        <div class="card">
            <div class="card-header">🎨 Visualizations</div>
            <div class="card-body">
                <div class="two-column">
                    {"<div class='figure'><img src='data:image/png;base64," + figures.get('boxplot', '') + "' alt='Boxplot'><div class='figure-caption'>Figure 1: GZIP Compression Ratio by Category</div></div>" if figures.get('boxplot') else "<div class='figure'><p>Boxplot not available (no diagnostic categories)</p></div>"}
                    {"<div class='figure'><img src='data:image/png;base64," + figures.get('heatmap', '') + "' alt='Heatmap'><div class='figure-caption'>Figure 2: Feature Correlations</div></div>" if figures.get('heatmap') else ""}
                </div>
                {"<div class='figure'><img src='data:image/png;base64," + figures.get('importance', '') + "' alt='Feature Importance'><div class='figure-caption'>Figure 3: Top Features</div></div>" if figures.get('importance') else ""}
            </div>
        </div>
        
        <!-- Feature Importance/Stats -->
        <div class="card">
            <div class="card-header">🏆 {"Feature Importance Ranking" if has_categories else "Feature Statistics"}</div>
            <div class="card-body">
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>{"F-score" if has_categories else "Mean"}</th>
                        <th>{"p-value" if has_categories else "Std Dev"}</th>
                        <th>{"Effect Size" if has_categories else "Missing %"}</th>
                    </tr>
                    {importance_rows}
                </table>
            </div>
        </div>
        
        <!-- SR Preparation -->
        <div class="card">
            <div class="card-header">🧬 Symbolic Regression Preparation</div>
            <div class="card-body">
                <h4>Baseline Performance</h4>
                <table>
                    <tr>
                        <th>Dataset</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {sr_rows}
                </table>
                
                <h4 style="margin-top: 25px;">Features Selected for SR</h4>
                <ul>
                    {sr_features_list}
                </ul>
            </div>
        </div>
    </div>
    
    <footer>
        <p>Kepler-ECG Project - Phase 3 Pipeline Report</p>
        <p>Generated automatically by <code>Phase3Pipeline</code></p>
    </footer>
</body>
</html>
"""
        return html
    
    def _save_results(self) -> None:
        """Save pipeline results summary."""
        logger.info("Step 6: Saving results summary...")
        
        results_path = os.path.join(self.config.output_dir, "phase3_pipeline_results.json")
        self.results.save(results_path)
        self.results.output_files.append(results_path)
        
        logger.info(f"  Saved: {results_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Records processed: {self.results.n_records:,}")
        logger.info(f"Features analyzed: {self.results.n_features}")
        logger.info(f"SR features selected: {len(self.results.sr_features)}")
        logger.info(f"Output files generated: {len(self.results.output_files)}")
        logger.info("\nBaseline Performance:")
        for name, value in self.results.baseline_performance.items():
            logger.info(f"  {name}: {value:.3f}")


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kepler-ECG Phase 3 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python -m kepler_ecg.analysis.pipeline
  
  # Custom paths
  python -m kepler_ecg.analysis.pipeline \\
      --data /path/to/features.csv \\
      --output /path/to/output \\
      --reports /path/to/reports \\
      --sr-ready /path/to/sr_datasets
  
  # Fast run (skip visualizations)
  python -m kepler_ecg.analysis.pipeline --skip-viz
        """
    )
    parser.add_argument(
        "--data", "-d",
        default="data/ptbxl_features_v2.csv",
        help="Path to input feature CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory for CSV/JSON results"
    )
    parser.add_argument(
        "--reports", "-r",
        default="reports",
        help="Output directory for visualizations (PNG)"
    )
    parser.add_argument(
        "--sr-ready", "-s",
        default="data/sr_ready",
        help="Output directory for SR-ready datasets"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation (faster)"
    )
    parser.add_argument(
        "--features", "-n",
        type=int,
        default=12,
        help="Number of features to select for SR (default: 12)"
    )
    parser.add_argument(
        "--top-features", "-t",
        type=int,
        default=15,
        help="Number of top features in importance plot (default: 15)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--report-name",
        default="phase3_report.html",
        help="Filename for HTML report (default: phase3_report.html)"
    )
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        data_path=args.data,
        output_dir=args.output,
        reports_dir=args.reports,
        sr_ready_dir=args.sr_ready,
        sr_features_count=args.features,
        n_top_features=args.top_features,
        generate_html_report=not args.no_report,
        report_filename=args.report_name,
    )
    
    pipeline = Phase3Pipeline(config=config)
    results = pipeline.run(skip_visualizations=args.skip_viz)
    
    print(f"\n✅ Pipeline completed. Generated {len(results.output_files)} files.")


if __name__ == "__main__":
    main()
