"""
Feature Selection Module for Kepler-ECG Phase 3.

This module provides comprehensive feature selection and importance analysis
tools for identifying the most discriminative ECG features, including:
- Univariate importance (ANOVA F-score, mutual information)
- Multivariate importance (Random Forest, Gradient Boosting)
- Redundancy analysis (correlation-based)
- Feature selection for symbolic regression

All methods support both classification and regression targets.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    mutual_info_regression,
    f_regression,
    SelectKBest,
    RFE,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""
    feature: str
    importance_score: float
    rank: int
    method: str
    p_value: Optional[float] = None
    
    def __str__(self) -> str:
        p_str = f", p={self.p_value:.2e}" if self.p_value is not None else ""
        return f"{self.rank}. {self.feature}: {self.importance_score:.4f} ({self.method}{p_str})"


@dataclass 
class RedundancyPair:
    """A pair of redundant features."""
    feature1: str
    feature2: str
    correlation: float
    recommendation: str  # Which feature to keep
    
    def __str__(self) -> str:
        return f"{self.feature1} <-> {self.feature2}: r={self.correlation:.3f} (keep: {self.recommendation})"


class FeatureSelector:
    """
    Feature selection and importance analysis for ECG data.
    
    Provides methods for:
    - Univariate feature importance (F-score, mutual information)
    - Multivariate importance (tree-based models)
    - Redundancy detection and removal
    - Optimal feature subset selection
    
    Example:
        >>> selector = FeatureSelector()
        >>> importance = selector.univariate_importance(df, 'diag_primary_category')
        >>> top_features = selector.select_top_features(df, 'diag_primary_category', n=20)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the feature selector.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Columns to exclude from feature analysis
        self.exclude_columns = {
            "ecg_id", "success", "processing_time_ms", "scp_codes",
            "quality_level", "is_usable", "diag_primary_code",
            "diag_primary_category", "diag_is_normal", "diag_is_multi_label",
            "diag_n_diagnoses", "hrv_hrv_spectral_method",
            "wav_wavelet_n_levels", "wav_wavelet_signal_length",
        }
    
    def _get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in self.exclude_columns]
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Prepare data for feature selection.
        
        Returns:
            X: Feature DataFrame
            y: Target array (encoded if categorical)
            feature_names: List of feature names
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Filter categories
        data = df[~df[target_col].isin(exclude_categories)].copy()
        
        # Get features
        if features is None:
            features = self._get_numeric_features(data)
        
        # Filter features with too many missing values (>50%)
        valid_features = []
        for f in features:
            if f in data.columns:
                missing_pct = data[f].isna().sum() / len(data)
                if missing_pct < 0.5:
                    valid_features.append(f)
        features = valid_features
        
        if len(features) == 0:
            raise ValueError("No valid features with <50% missing values")
        
        # Prepare X and y
        X = data[features].copy()
        y = data[target_col].values
        
        # Handle missing values - drop rows with any NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No samples remaining after dropping NaN values")
        
        # Encode categorical target if needed
        if y.dtype == object or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y, features
    
    def univariate_importance(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        features: Optional[List[str]] = None,
        methods: List[str] = ["f_score", "mutual_info"],
        exclude_categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate univariate feature importance using multiple methods.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            features: List of features (if None, use all numeric)
            methods: List of methods to use ('f_score', 'mutual_info')
            exclude_categories: Categories to exclude
            
        Returns:
            DataFrame with feature importance scores and rankings
        """
        X, y, feature_names = self._prepare_data(
            df, target_col, features, exclude_categories
        )
        
        results = {"feature": feature_names}
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 20
        
        if "f_score" in methods:
            if is_classification:
                f_scores, p_values = f_classif(X, y)
            else:
                f_scores, p_values = f_regression(X, y)
            
            results["f_score"] = f_scores
            results["f_score_pvalue"] = p_values
        
        if "mutual_info" in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if is_classification:
                    mi_scores = mutual_info_classif(
                        X, y, random_state=self.random_state, n_neighbors=5
                    )
                else:
                    mi_scores = mutual_info_regression(
                        X, y, random_state=self.random_state, n_neighbors=5
                    )
            results["mutual_info"] = mi_scores
        
        # Create DataFrame
        result_df = pd.DataFrame(results)
        
        # Add rankings
        if "f_score" in methods:
            result_df["f_score_rank"] = result_df["f_score"].rank(ascending=False).astype(int)
        if "mutual_info" in methods:
            result_df["mutual_info_rank"] = result_df["mutual_info"].rank(ascending=False).astype(int)
        
        # Combined rank (average of available ranks)
        rank_cols = [c for c in result_df.columns if c.endswith("_rank")]
        if rank_cols:
            result_df["combined_rank"] = result_df[rank_cols].mean(axis=1).rank().astype(int)
        
        return result_df.sort_values("combined_rank").reset_index(drop=True)
    
    def multivariate_importance(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        features: Optional[List[str]] = None,
        method: str = "random_forest",
        exclude_categories: Optional[List[str]] = None,
        n_estimators: int = 100,
        use_permutation: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate feature importance using tree-based models.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            features: List of features
            method: 'random_forest' or 'gradient_boosting'
            exclude_categories: Categories to exclude
            n_estimators: Number of trees
            use_permutation: Use permutation importance (slower but more reliable)
            
        Returns:
            DataFrame with feature importance scores
        """
        X, y, feature_names = self._prepare_data(
            df, target_col, features, exclude_categories
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 20
        
        # Select model
        if method == "random_forest":
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_depth=10,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_depth=10,
                )
        elif method == "gradient_boosting":
            if is_classification:
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    max_depth=5,
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    max_depth=5,
                )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled, y)
        
        # Get importance
        if use_permutation:
            perm_importance = permutation_importance(
                model, X_scaled, y,
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            importance = perm_importance.importances_mean
            importance_std = perm_importance.importances_std
        else:
            importance = model.feature_importances_
            importance_std = np.zeros_like(importance)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            "feature": feature_names,
            f"{method}_importance": importance,
            f"{method}_importance_std": importance_std,
        })
        
        result_df[f"{method}_rank"] = result_df[f"{method}_importance"].rank(ascending=False).astype(int)
        
        return result_df.sort_values(f"{method}_rank").reset_index(drop=True)
    
    def combined_importance(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        features: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate combined feature importance using multiple methods.
        
        Combines:
        - ANOVA F-score
        - Mutual Information
        - Random Forest importance
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            features: List of features
            exclude_categories: Categories to exclude
            
        Returns:
            DataFrame with combined importance scores and final ranking
        """
        # Get univariate importance
        uni_importance = self.univariate_importance(
            df, target_col, features, 
            methods=["f_score", "mutual_info"],
            exclude_categories=exclude_categories,
        )
        
        # Get multivariate importance
        multi_importance = self.multivariate_importance(
            df, target_col, features,
            method="random_forest",
            exclude_categories=exclude_categories,
        )
        
        # Merge results
        combined = uni_importance.merge(
            multi_importance[["feature", "random_forest_importance", "random_forest_rank"]],
            on="feature",
            how="outer",
        )
        
        # Calculate combined score (normalized average)
        # Normalize each score to 0-1 range
        for col in ["f_score", "mutual_info", "random_forest_importance"]:
            if col in combined.columns:
                max_val = combined[col].max()
                if max_val > 0:
                    combined[f"{col}_norm"] = combined[col] / max_val
        
        norm_cols = [c for c in combined.columns if c.endswith("_norm")]
        if norm_cols:
            combined["combined_score"] = combined[norm_cols].mean(axis=1)
            combined["final_rank"] = combined["combined_score"].rank(ascending=False).astype(int)
        
        return combined.sort_values("final_rank").reset_index(drop=True)
    
    def select_top_features(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        n_features: int = 20,
        method: str = "combined",
        exclude_categories: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select top N most discriminative features.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            n_features: Number of features to select
            method: Selection method ('combined', 'f_score', 'mutual_info', 'random_forest')
            exclude_categories: Categories to exclude
            
        Returns:
            List of top feature names
        """
        if method == "combined":
            importance = self.combined_importance(
                df, target_col, exclude_categories=exclude_categories
            )
            return importance.head(n_features)["feature"].tolist()
        
        elif method in ["f_score", "mutual_info"]:
            importance = self.univariate_importance(
                df, target_col,
                methods=[method],
                exclude_categories=exclude_categories,
            )
            rank_col = f"{method}_rank"
            return importance.nsmallest(n_features, rank_col)["feature"].tolist()
        
        elif method == "random_forest":
            importance = self.multivariate_importance(
                df, target_col,
                method="random_forest",
                exclude_categories=exclude_categories,
            )
            return importance.head(n_features)["feature"].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def redundancy_analysis(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.9,
        method: str = "pearson",
    ) -> Tuple[pd.DataFrame, List[RedundancyPair]]:
        """
        Identify redundant features based on correlation.
        
        Args:
            df: DataFrame with features
            features: List of features to analyze
            threshold: Correlation threshold for redundancy
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Tuple of (correlation matrix, list of redundant pairs)
        """
        if features is None:
            features = self._get_numeric_features(df)
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr(method=method)
        
        # Find redundant pairs
        redundant_pairs = []
        seen = set()
        
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if i >= j:
                    continue
                
                corr = abs(corr_matrix.loc[f1, f2])
                if corr >= threshold:
                    # Determine which to keep (prefer simpler name or first alphabetically)
                    keep = f1 if len(f1) <= len(f2) else f2
                    
                    pair = RedundancyPair(
                        feature1=f1,
                        feature2=f2,
                        correlation=corr_matrix.loc[f1, f2],
                        recommendation=keep,
                    )
                    redundant_pairs.append(pair)
        
        # Sort by correlation strength
        redundant_pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return corr_matrix, redundant_pairs
    
    def remove_redundant_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str = "diag_primary_category",
        threshold: float = 0.9,
        exclude_categories: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Remove redundant features, keeping the most informative one.
        
        For each pair of highly correlated features, keeps the one
        with higher importance for the target.
        
        Args:
            df: DataFrame with features
            features: List of features to filter
            target_col: Target column for importance comparison
            threshold: Correlation threshold
            exclude_categories: Categories to exclude
            
        Returns:
            List of non-redundant features
        """
        # Get importance scores
        importance = self.univariate_importance(
            df, target_col, features,
            methods=["f_score"],
            exclude_categories=exclude_categories,
        )
        importance_dict = dict(zip(importance["feature"], importance["f_score"]))
        
        # Get redundant pairs
        _, redundant_pairs = self.redundancy_analysis(df, features, threshold)
        
        # Track features to remove
        to_remove = set()
        
        for pair in redundant_pairs:
            if pair.feature1 in to_remove or pair.feature2 in to_remove:
                continue
            
            # Keep the more important feature
            imp1 = importance_dict.get(pair.feature1, 0)
            imp2 = importance_dict.get(pair.feature2, 0)
            
            if imp1 >= imp2:
                to_remove.add(pair.feature2)
            else:
                to_remove.add(pair.feature1)
        
        return [f for f in features if f not in to_remove]
    
    def get_feature_groups(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Group features by their prefix/category.
        
        Args:
            df: DataFrame with features
            features: List of features to group
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        if features is None:
            features = self._get_numeric_features(df)
        
        groups = {
            "compressibility_signal": [],
            "compressibility_rr": [],
            "wavelet": [],
            "hrv": [],
            "morphology": [],
            "intervals": [],
            "basic": [],
            "other": [],
        }
        
        for f in features:
            if f.startswith("comp_sig_"):
                groups["compressibility_signal"].append(f)
            elif f.startswith("comp_rr_"):
                groups["compressibility_rr"].append(f)
            elif f.startswith("wav_"):
                groups["wavelet"].append(f)
            elif f.startswith("hrv_"):
                groups["hrv"].append(f)
            elif f.startswith("morph_"):
                groups["morphology"].append(f)
            elif f.startswith("interval_"):
                groups["intervals"].append(f)
            elif f in ["heart_rate_bpm", "heart_rate_std", "rr_mean_ms", 
                       "rr_std_ms", "rmssd", "pnn50", "n_beats", "age", "sex"]:
                groups["basic"].append(f)
            else:
                groups["other"].append(f)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def analyze_feature_by_group(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze feature importance by feature group.
        
        Args:
            df: DataFrame with features
            target_col: Target column
            exclude_categories: Categories to exclude
            
        Returns:
            DataFrame with group-level statistics
        """
        # Get all feature importance
        importance = self.combined_importance(
            df, target_col, exclude_categories=exclude_categories
        )
        
        # Get feature groups
        groups = self.get_feature_groups(df)
        
        # Calculate group statistics
        group_stats = []
        for group_name, group_features in groups.items():
            group_imp = importance[importance["feature"].isin(group_features)]
            
            if len(group_imp) > 0:
                group_stats.append({
                    "group": group_name,
                    "n_features": len(group_imp),
                    "mean_f_score": group_imp["f_score"].mean(),
                    "max_f_score": group_imp["f_score"].max(),
                    "mean_combined_score": group_imp["combined_score"].mean() if "combined_score" in group_imp else 0,
                    "best_feature": group_imp.iloc[0]["feature"] if len(group_imp) > 0 else None,
                    "top_3_features": group_imp.head(3)["feature"].tolist(),
                })
        
        return pd.DataFrame(group_stats).sort_values("mean_f_score", ascending=False)
    
    def suggest_features_for_sr(
        self,
        df: pd.DataFrame,
        target_col: str = "diag_primary_category",
        n_features: int = 10,
        max_correlation: float = 0.8,
        exclude_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest optimal feature set for Symbolic Regression.
        
        Selects features that are:
        - Highly discriminative
        - Not redundant (low inter-correlation)
        - From diverse feature groups
        
        Args:
            df: DataFrame with features
            target_col: Target column
            n_features: Number of features to suggest
            max_correlation: Maximum allowed correlation between features
            exclude_categories: Categories to exclude
            
        Returns:
            Dictionary with suggested features and metadata
        """
        # Get combined importance
        importance = self.combined_importance(
            df, target_col, exclude_categories=exclude_categories
        )
        
        # Get all features sorted by importance
        all_features = importance["feature"].tolist()
        
        # Remove redundant features iteratively
        selected = []
        
        for feature in all_features:
            if len(selected) >= n_features:
                break
            
            # Check if this feature is too correlated with already selected
            if len(selected) == 0:
                selected.append(feature)
                continue
            
            # Calculate correlation with selected features
            subset = df[[feature] + selected].dropna()
            if len(subset) < 100:
                continue
            
            correlations = subset[selected].corrwith(subset[feature]).abs()
            
            if correlations.max() < max_correlation:
                selected.append(feature)
        
        # Get importance info for selected features
        selected_importance = importance[importance["feature"].isin(selected)]
        
        # Get group distribution
        groups = self.get_feature_groups(df, selected)
        
        return {
            "suggested_features": selected,
            "n_features": len(selected),
            "importance_summary": selected_importance[["feature", "f_score", "combined_score", "final_rank"]].to_dict("records"),
            "group_distribution": {k: len(v) for k, v in groups.items()},
            "max_pairwise_correlation": max_correlation,
        }
