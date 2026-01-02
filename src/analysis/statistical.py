"""
Statistical Analysis Module for Kepler-ECG Phase 3.

This module provides comprehensive statistical analysis tools for comparing
ECG features across diagnostic categories, including ANOVA, post-hoc tests,
effect size calculations, and correlation analysis.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, levene


@dataclass
class ANOVAResult:
    """Results from ANOVA analysis."""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    n_groups: int
    n_total: int
    is_significant: bool
    significance_level: float = 0.05
    
    def __str__(self) -> str:
        sig_str = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        return (
            f"ANOVA Results:\n"
            f"  F({self.df_between}, {self.df_within}) = {self.f_statistic:.4f}\n"
            f"  p-value = {self.p_value:.2e} {sig_str}\n"
            f"  Groups: {self.n_groups}, Total N: {self.n_total}"
        )


@dataclass
class PairwiseComparison:
    """Results from a single pairwise comparison."""
    group1: str
    group2: str
    mean_diff: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    
    @property
    def effect_size_label(self) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def __str__(self) -> str:
        sig_str = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        return (
            f"{self.group1} vs {self.group2}: "
            f"diff={self.mean_diff:+.4f}, p={self.p_value:.2e} {sig_str}, "
            f"d={self.cohens_d:.3f} ({self.effect_size_label})"
        )


@dataclass
class CategoryComparisonResult:
    """Complete results from category comparison analysis."""
    feature: str
    category_column: str
    anova: ANOVAResult
    pairwise: List[PairwiseComparison]
    descriptive: pd.DataFrame
    normality_test: Dict[str, Tuple[float, float]]  # group -> (statistic, p-value)
    homogeneity_test: Tuple[float, float]  # Levene's test (statistic, p-value)
    used_nonparametric: bool = False
    
    def get_significant_pairs(self) -> List[PairwiseComparison]:
        """Return only significant pairwise comparisons."""
        return [p for p in self.pairwise if p.is_significant]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "category_column": self.category_column,
            "anova": {
                "f_statistic": self.anova.f_statistic,
                "p_value": self.anova.p_value,
                "df_between": self.anova.df_between,
                "df_within": self.anova.df_within,
                "is_significant": self.anova.is_significant,
            },
            "pairwise": [
                {
                    "group1": p.group1,
                    "group2": p.group2,
                    "mean_diff": p.mean_diff,
                    "p_value": p.p_value,
                    "cohens_d": p.cohens_d,
                    "effect_size": p.effect_size_label,
                    "is_significant": p.is_significant,
                }
                for p in self.pairwise
            ],
            "descriptive": self.descriptive.to_dict(),
            "used_nonparametric": self.used_nonparametric,
        }


class StatisticalAnalyzer:
    """
    Statistical analysis tools for ECG feature comparison across diagnostic categories.
    
    This class provides methods for:
    - ANOVA and Kruskal-Wallis tests for group comparisons
    - Post-hoc pairwise comparisons with multiple testing correction
    - Effect size calculations (Cohen's d)
    - Correlation analysis
    - Feature-category association analysis
    
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> result = analyzer.compare_categories(df, 'comp_rr_approx_entropy')
        >>> print(result.anova)
        >>> for pair in result.get_significant_pairs():
        ...     print(pair)
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        normality_threshold: float = 0.05,
        min_group_size: int = 30,
    ):
        """
        Initialize the statistical analyzer.
        
        Args:
            significance_level: Alpha level for significance testing
            normality_threshold: P-value threshold for normality tests
            min_group_size: Minimum samples per group for analysis
        """
        self.significance_level = significance_level
        self.normality_threshold = normality_threshold
        self.min_group_size = min_group_size
    
    def compare_categories(
        self,
        df: pd.DataFrame,
        feature: str,
        category_col: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
    ) -> CategoryComparisonResult:
        """
        Compare a feature across diagnostic categories.
        
        Performs ANOVA (or Kruskal-Wallis if assumptions violated),
        post-hoc pairwise comparisons, and effect size calculations.
        
        Args:
            df: DataFrame with feature and category columns
            feature: Name of the feature column to analyze
            category_col: Name of the category column
            exclude_categories: Categories to exclude (e.g., ['UNKNOWN'])
            
        Returns:
            CategoryComparisonResult with all statistical results
            
        Raises:
            ValueError: If feature or category column not found
        """
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
        if category_col not in df.columns:
            raise ValueError(f"Category column '{category_col}' not found in DataFrame")
        
        # Filter data
        data = df[[feature, category_col]].dropna()
        
        if exclude_categories:
            data = data[~data[category_col].isin(exclude_categories)]
        
        # Get groups
        groups = data[category_col].unique()
        group_data = {g: data[data[category_col] == g][feature].values for g in groups}
        
        # Filter out small groups
        group_data = {g: v for g, v in group_data.items() if len(v) >= self.min_group_size}
        groups = list(group_data.keys())
        
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups with >= {self.min_group_size} samples")
        
        # Test assumptions
        normality_results = self._test_normality(group_data)
        homogeneity_stat, homogeneity_p = self._test_homogeneity(list(group_data.values()))
        
        # Decide parametric vs non-parametric
        normality_violated = any(p < self.normality_threshold for _, p in normality_results.values())
        homogeneity_violated = homogeneity_p < self.normality_threshold
        use_nonparametric = normality_violated or homogeneity_violated
        
        # Run main test
        if use_nonparametric:
            anova_result = self._kruskal_wallis(group_data)
        else:
            anova_result = self._anova(group_data)
        
        # Post-hoc pairwise comparisons
        pairwise_results = self._pairwise_comparisons(group_data, use_nonparametric)
        
        # Descriptive statistics
        descriptive = self._descriptive_stats(group_data)
        
        return CategoryComparisonResult(
            feature=feature,
            category_column=category_col,
            anova=anova_result,
            pairwise=pairwise_results,
            descriptive=descriptive,
            normality_test=normality_results,
            homogeneity_test=(homogeneity_stat, homogeneity_p),
            used_nonparametric=use_nonparametric,
        )
    
    def _test_normality(self, group_data: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Test normality for each group using Shapiro-Wilk test."""
        results = {}
        for group, values in group_data.items():
            # Shapiro-Wilk works best with n < 5000, subsample if needed
            if len(values) > 5000:
                sample = np.random.choice(values, 5000, replace=False)
            else:
                sample = values
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = shapiro(sample)
            results[group] = (stat, p)
        return results
    
    def _test_homogeneity(self, groups: List[np.ndarray]) -> Tuple[float, float]:
        """Test homogeneity of variances using Levene's test."""
        stat, p = levene(*groups)
        return stat, p
    
    def _anova(self, group_data: Dict[str, np.ndarray]) -> ANOVAResult:
        """Perform one-way ANOVA."""
        groups = list(group_data.values())
        f_stat, p_value = f_oneway(*groups)
        
        n_groups = len(groups)
        n_total = sum(len(g) for g in groups)
        df_between = n_groups - 1
        df_within = n_total - n_groups
        
        return ANOVAResult(
            f_statistic=f_stat,
            p_value=p_value,
            df_between=df_between,
            df_within=df_within,
            n_groups=n_groups,
            n_total=n_total,
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )
    
    def _kruskal_wallis(self, group_data: Dict[str, np.ndarray]) -> ANOVAResult:
        """Perform Kruskal-Wallis H-test (non-parametric ANOVA)."""
        groups = list(group_data.values())
        h_stat, p_value = kruskal(*groups)
        
        n_groups = len(groups)
        n_total = sum(len(g) for g in groups)
        
        return ANOVAResult(
            f_statistic=h_stat,  # H-statistic stored as F for consistency
            p_value=p_value,
            df_between=n_groups - 1,
            df_within=n_total - n_groups,
            n_groups=n_groups,
            n_total=n_total,
            is_significant=p_value < self.significance_level,
            significance_level=self.significance_level,
        )
    
    def _pairwise_comparisons(
        self,
        group_data: Dict[str, np.ndarray],
        use_nonparametric: bool = False,
    ) -> List[PairwiseComparison]:
        """
        Perform pairwise comparisons with Bonferroni correction.
        """
        groups = list(group_data.keys())
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        adjusted_alpha = self.significance_level / n_comparisons
        
        results = []
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                v1, v2 = group_data[g1], group_data[g2]
                
                # Statistical test
                if use_nonparametric:
                    _, p_value = mannwhitneyu(v1, v2, alternative='two-sided')
                else:
                    _, p_value = stats.ttest_ind(v1, v2)
                
                # Effect size (Cohen's d)
                cohens_d = self._cohens_d(v1, v2)
                
                # Mean difference
                mean_diff = np.mean(v1) - np.mean(v2)
                
                # Confidence interval for mean difference
                ci_lower, ci_upper = self._mean_diff_ci(v1, v2)
                
                results.append(PairwiseComparison(
                    group1=g1,
                    group2=g2,
                    mean_diff=mean_diff,
                    p_value=p_value,
                    cohens_d=cohens_d,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    is_significant=p_value < adjusted_alpha,
                ))
        
        return sorted(results, key=lambda x: x.p_value)
    
    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def _mean_diff_ci(
        group1: np.ndarray,
        group2: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference."""
        mean_diff = np.mean(group1) - np.mean(group2)
        se = np.sqrt(np.var(group1, ddof=1) / len(group1) + np.var(group2, ddof=1) / len(group2))
        
        df = len(group1) + len(group2) - 2
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        
        margin = t_crit * se
        return mean_diff - margin, mean_diff + margin
    
    def _descriptive_stats(self, group_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate descriptive statistics for each group."""
        stats_list = []
        for group, values in group_data.items():
            stats_list.append({
                "group": group,
                "n": len(values),
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "iqr": np.percentile(values, 75) - np.percentile(values, 25),
                "skewness": stats.skew(values),
                "kurtosis": stats.kurtosis(values),
            })
        
        return pd.DataFrame(stats_list).set_index("group").sort_values("mean")
    
    def correlation_matrix(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        feature_prefix: Optional[str] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for a set of features.
        
        Args:
            df: DataFrame with features
            features: List of feature names (if None, use feature_prefix)
            feature_prefix: Prefix to filter features (e.g., 'comp_')
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        if features is None and feature_prefix is None:
            raise ValueError("Must provide either features list or feature_prefix")
        
        if features is None:
            features = [c for c in df.columns if c.startswith(feature_prefix)]
        
        return df[features].corr(method=method)
    
    def feature_category_association(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        category_col: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate association between each feature and diagnostic categories.
        
        Args:
            df: DataFrame with features and categories
            features: List of features to analyze (if None, infer numeric columns)
            category_col: Category column name
            exclude_categories: Categories to exclude
            
        Returns:
            DataFrame with feature, F-statistic, p-value, effect_size for each feature
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        if features is None:
            # Get numeric features, excluding metadata columns
            exclude_cols = {
                "ecg_id", "success", "processing_time_ms", "scp_codes",
                "quality_level", "is_usable", "diag_primary_code",
                "diag_primary_category", "diag_is_normal", "diag_is_multi_label",
                "hrv_hrv_spectral_method", "wav_wavelet_n_levels", "wav_wavelet_signal_length",
            }
            features = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols
            ]
        
        results = []
        for feature in features:
            try:
                result = self.compare_categories(
                    df, feature, category_col, exclude_categories
                )
                
                # Calculate eta-squared (effect size for ANOVA)
                # Find maximum effect size among pairwise comparisons
                max_effect = max(abs(p.cohens_d) for p in result.pairwise) if result.pairwise else 0
                
                results.append({
                    "feature": feature,
                    "f_statistic": result.anova.f_statistic,
                    "p_value": result.anova.p_value,
                    "max_cohens_d": max_effect,
                    "is_significant": result.anova.is_significant,
                    "n_significant_pairs": len(result.get_significant_pairs()),
                    "used_nonparametric": result.used_nonparametric,
                })
            except Exception as e:
                # Skip features that can't be analyzed (too many NaN, etc.)
                continue
        
        return (
            pd.DataFrame(results)
            .sort_values("p_value")
            .reset_index(drop=True)
        )
    
    def compare_two_groups(
        self,
        df: pd.DataFrame,
        feature: str,
        group1: str,
        group2: str,
        category_col: str = "diag_primary_category",
    ) -> Dict[str, Any]:
        """
        Compare a feature between exactly two groups.
        
        Useful for focused comparisons like NORM vs HYP.
        
        Args:
            df: DataFrame with feature and category
            feature: Feature column name
            group1: First group name (e.g., 'NORM')
            group2: Second group name (e.g., 'HYP')
            category_col: Category column name
            
        Returns:
            Dictionary with test results, effect size, and descriptive stats
        """
        data = df[[feature, category_col]].dropna()
        v1 = data[data[category_col] == group1][feature].values
        v2 = data[data[category_col] == group2][feature].values
        
        if len(v1) < self.min_group_size or len(v2) < self.min_group_size:
            raise ValueError(f"Insufficient samples: {group1}={len(v1)}, {group2}={len(v2)}")
        
        # T-test and Mann-Whitney
        t_stat, t_pvalue = stats.ttest_ind(v1, v2)
        u_stat, u_pvalue = mannwhitneyu(v1, v2, alternative='two-sided')
        
        # Effect size
        cohens_d = self._cohens_d(v1, v2)
        
        return {
            "feature": feature,
            "group1": group1,
            "group2": group2,
            "n1": len(v1),
            "n2": len(v2),
            "mean1": np.mean(v1),
            "mean2": np.mean(v2),
            "std1": np.std(v1, ddof=1),
            "std2": np.std(v2, ddof=1),
            "mean_diff": np.mean(v1) - np.mean(v2),
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "u_statistic": u_stat,
            "u_pvalue": u_pvalue,
            "cohens_d": cohens_d,
            "effect_size_label": (
                "negligible" if abs(cohens_d) < 0.2 else
                "small" if abs(cohens_d) < 0.5 else
                "medium" if abs(cohens_d) < 0.8 else
                "large"
            ),
        }
