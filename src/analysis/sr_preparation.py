"""
Symbolic Regression Preparation Module for Kepler-ECG Phase 3.

This module prepares ECG feature data for symbolic regression using PySR,
including:
- Binary classification datasets (NORM vs specific pathology)
- Multi-class classification datasets
- Regression datasets (e.g., cardiac age prediction)
- Data normalization and cleaning
- Primitive operator suggestions

The goal is to discover interpretable mathematical formulas that relate
ECG features to diagnostic categories or continuous outcomes.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


@dataclass
class SRDataset:
    """A dataset prepared for symbolic regression."""
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_name: str
    n_samples: int
    n_features: int
    task_type: str  # 'binary', 'multiclass', 'regression'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with features and target."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df[self.target_name] = self.y
        return df
    
    def save(self, path: str) -> None:
        """Save dataset to CSV."""
        self.to_dataframe().to_csv(path, index=False)
    
    def __str__(self) -> str:
        return (
            f"SRDataset(task={self.task_type}, "
            f"samples={self.n_samples}, features={self.n_features}, "
            f"target={self.target_name})"
        )


@dataclass
class SRPrimitives:
    """Suggested primitives for symbolic regression."""
    binary_operators: List[str]
    unary_operators: List[str]
    constraints: Dict[str, Any]
    complexity_weights: Dict[str, float]
    rationale: str
    
    def to_pysr_config(self) -> Dict[str, Any]:
        """Convert to PySR configuration format."""
        return {
            "binary_operators": self.binary_operators,
            "unary_operators": self.unary_operators,
            "complexity_of_operators": self.complexity_weights,
        }
    
    def __str__(self) -> str:
        return (
            f"Binary: {self.binary_operators}\n"
            f"Unary: {self.unary_operators}\n"
            f"Rationale: {self.rationale}"
        )


class SymbolicRegressionPrep:
    """
    Prepare data for symbolic regression with PySR.
    
    Provides methods for:
    - Creating binary classification datasets (e.g., NORM vs MI)
    - Creating regression datasets (e.g., predicting age)
    - Normalizing and cleaning data
    - Suggesting appropriate mathematical operators
    
    Example:
        >>> prep = SymbolicRegressionPrep()
        >>> dataset = prep.prepare_binary_classification(df, 'HYP', 'NORM')
        >>> dataset.save('norm_vs_hyp.csv')
        >>> primitives = prep.suggest_sr_primitives(df, dataset.feature_names)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
    ):
        """
        Initialize the SR preparation module.
        
        Args:
            random_state: Random seed for reproducibility
            test_size: Fraction of data to hold out for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        
        # Default features to exclude
        self.exclude_columns = {
            "ecg_id", "success", "processing_time_ms", "scp_codes",
            "quality_level", "is_usable", "diag_primary_code",
            "diag_primary_category", "diag_is_normal", "diag_is_multi_label",
            "diag_n_diagnoses", "hrv_hrv_spectral_method",
            "wav_wavelet_n_levels", "wav_wavelet_signal_length",
        }
    
    def _get_numeric_features(
        self,
        df: pd.DataFrame,
        exclude_target: Optional[str] = None,
    ) -> List[str]:
        """Get list of numeric feature columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = self.exclude_columns.copy()
        if exclude_target:
            exclude.add(exclude_target)
        return [c for c in numeric_cols if c not in exclude]
    
    def prepare_binary_classification(
        self,
        df: pd.DataFrame,
        positive_class: str,
        negative_class: str = "NORM",
        category_col: str = "diag_primary_category",
        features: Optional[List[str]] = None,
        normalize: bool = True,
        balance_classes: bool = True,
    ) -> SRDataset:
        """
        Prepare dataset for binary classification.
        
        Creates a dataset for distinguishing between two diagnostic categories,
        typically NORM vs a specific pathology.
        
        Args:
            df: DataFrame with features and diagnosis
            positive_class: Positive class label (e.g., 'MI', 'HYP')
            negative_class: Negative class label (default: 'NORM')
            category_col: Column containing diagnostic categories
            features: List of features to include (None for auto-select)
            normalize: Whether to normalize features
            balance_classes: Whether to balance class sizes
            
        Returns:
            SRDataset ready for symbolic regression
        """
        # Filter to two classes
        mask = df[category_col].isin([positive_class, negative_class])
        data = df[mask].copy()
        
        # Get features
        if features is None:
            features = self._get_numeric_features(data, category_col)
        
        # Filter features with too many missing values
        valid_features = []
        for f in features:
            if f in data.columns:
                missing_pct = data[f].isna().sum() / len(data)
                if missing_pct < 0.3:  # Max 30% missing
                    valid_features.append(f)
        features = valid_features
        
        # Prepare X and y
        X = data[features].copy()
        y = (data[category_col] == positive_class).astype(int).values
        
        # Drop rows with NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        # Balance classes if requested
        if balance_classes:
            X, y = self._balance_classes(X.values, y)
        else:
            X = X.values
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return SRDataset(
            X=X,
            y=y,
            feature_names=features,
            target_name=f"{positive_class}_vs_{negative_class}",
            n_samples=len(X),
            n_features=len(features),
            task_type="binary",
            metadata={
                "positive_class": positive_class,
                "negative_class": negative_class,
                "normalized": normalize,
                "balanced": balance_classes,
                "positive_count": int(y.sum()),
                "negative_count": int(len(y) - y.sum()),
            },
        )
    
    def prepare_multiclass(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None,
        category_col: str = "diag_primary_category",
        features: Optional[List[str]] = None,
        normalize: bool = True,
        exclude_categories: Optional[List[str]] = None,
    ) -> SRDataset:
        """
        Prepare dataset for multi-class classification.
        
        Args:
            df: DataFrame with features and diagnosis
            categories: List of categories to include (None for all)
            category_col: Column containing diagnostic categories
            features: List of features to include
            normalize: Whether to normalize features
            exclude_categories: Categories to exclude
            
        Returns:
            SRDataset for multi-class classification
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Filter categories
        data = df[~df[category_col].isin(exclude_categories)].copy()
        
        if categories is not None:
            data = data[data[category_col].isin(categories)]
        
        # Get features
        if features is None:
            features = self._get_numeric_features(data, category_col)
        
        # Filter features with missing values
        valid_features = [f for f in features 
                         if f in data.columns and data[f].isna().sum() / len(data) < 0.3]
        features = valid_features
        
        # Prepare X and y
        X = data[features].copy()
        
        # Encode categories
        le = LabelEncoder()
        y = le.fit_transform(data[category_col].values)
        
        # Drop rows with NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask].values
        y = y[mask]
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return SRDataset(
            X=X,
            y=y,
            feature_names=features,
            target_name="diagnosis_class",
            n_samples=len(X),
            n_features=len(features),
            task_type="multiclass",
            metadata={
                "classes": list(le.classes_),
                "class_mapping": {c: i for i, c in enumerate(le.classes_)},
                "normalized": normalize,
                "class_counts": {c: int((y == i).sum()) for i, c in enumerate(le.classes_)},
            },
        )
    
    def prepare_regression_target(
        self,
        df: pd.DataFrame,
        target: str = "age",
        features: Optional[List[str]] = None,
        normalize_features: bool = True,
        normalize_target: bool = False,
        exclude_categories: Optional[List[str]] = None,
        filter_category: Optional[str] = None,
    ) -> SRDataset:
        """
        Prepare dataset for regression (e.g., cardiac age prediction).
        
        This is for Stream B: discovering formulas that predict continuous
        variables like age from ECG features.
        
        Args:
            df: DataFrame with features and target
            target: Target column name (e.g., 'age')
            features: List of features to include
            normalize_features: Whether to normalize features
            normalize_target: Whether to normalize target
            exclude_categories: Categories to exclude
            filter_category: Only include this category (e.g., 'NORM')
            
        Returns:
            SRDataset for regression
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Filter data
        data = df[~df["diag_primary_category"].isin(exclude_categories)].copy()
        
        if filter_category:
            data = data[data["diag_primary_category"] == filter_category]
        
        # Get features
        if features is None:
            features = self._get_numeric_features(data, target)
        
        # Remove target from features if present
        features = [f for f in features if f != target]
        
        # Filter features with missing values
        valid_features = [f for f in features 
                         if f in data.columns and data[f].isna().sum() / len(data) < 0.3]
        features = valid_features
        
        # Prepare X and y
        X = data[features].copy()
        y = data[target].values
        
        # Drop rows with NaN
        mask = ~X.isna().any(axis=1) & ~np.isnan(y)
        X = X[mask].values
        y = y[mask]
        
        # Store original target stats for denormalization
        target_mean = y.mean()
        target_std = y.std()
        
        # Normalize
        if normalize_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        if normalize_target:
            y = (y - target_mean) / target_std
        
        return SRDataset(
            X=X,
            y=y,
            feature_names=features,
            target_name=target,
            n_samples=len(X),
            n_features=len(features),
            task_type="regression",
            metadata={
                "target_mean": float(target_mean),
                "target_std": float(target_std),
                "target_min": float(y.min()),
                "target_max": float(y.max()),
                "normalized_features": normalize_features,
                "normalized_target": normalize_target,
                "filter_category": filter_category,
            },
        )
    
    def create_normalized_dataset(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: Optional[str] = None,
        method: str = "zscore",
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Create a normalized dataset for SR.
        
        Args:
            df: Input DataFrame
            features: Features to include
            target_col: Optional target column to include
            method: Normalization method ('zscore', 'minmax')
            remove_outliers: Whether to remove extreme outliers
            outlier_threshold: Z-score threshold for outlier removal
            
        Returns:
            Normalized DataFrame
        """
        # Select columns
        cols = features.copy()
        if target_col:
            cols.append(target_col)
        
        data = df[cols].copy()
        
        # Drop rows with NaN
        data = data.dropna()
        
        # Remove outliers
        if remove_outliers:
            for f in features:
                z_scores = np.abs(stats.zscore(data[f]))
                data = data[z_scores < outlier_threshold]
        
        # Normalize features
        if method == "zscore":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        data[features] = scaler.fit_transform(data[features])
        
        return data
    
    def suggest_sr_primitives(
        self,
        df: pd.DataFrame,
        features: List[str],
        task_type: str = "classification",
    ) -> SRPrimitives:
        """
        Suggest mathematical operators for PySR based on feature distributions.
        
        Analyzes the data to recommend appropriate primitive operations
        for symbolic regression.
        
        Args:
            df: DataFrame with features
            features: List of features to analyze
            task_type: 'classification' or 'regression'
            
        Returns:
            SRPrimitives with recommended operators
        """
        # Analyze feature distributions
        data = df[features].dropna()
        
        # Check for different data characteristics
        has_negative = (data < 0).any().any()
        has_near_zero = (data.abs() < 0.01).any().any()
        is_bounded = ((data >= 0) & (data <= 1)).all().all()
        has_high_variance = (data.std() > 10).any()
        
        # Determine skewness
        skewness = data.skew().abs().mean()
        is_skewed = skewness > 1.0
        
        # Build operator recommendations
        binary_operators = ["+", "-", "*"]
        unary_operators = ["square", "sqrt", "abs"]
        
        # Add division if no near-zero values
        if not has_near_zero:
            binary_operators.append("/")
        
        # Add log/exp for skewed data (but protect log from negatives)
        if is_skewed and not has_negative:
            unary_operators.extend(["log", "exp"])
        elif is_skewed:
            unary_operators.append("exp")
        
        # Add trigonometric for periodic patterns (rare in ECG summary stats)
        # unary_operators.extend(["sin", "cos"])
        
        # Add cube for non-linear relationships
        if has_high_variance:
            unary_operators.append("cube")
        
        # Constraints
        constraints = {
            "max_depth": 8,
            "max_size": 30,
            "nested_constraints": {
                "sqrt": {"sqrt": 0, "log": 0},  # Prevent sqrt(sqrt(...)) 
                "log": {"log": 0, "sqrt": 0},
                "exp": {"exp": 0},  # Prevent exp(exp(...))
            },
        }
        
        # Complexity weights (higher = more penalized)
        complexity_weights = {
            "+": 1, "-": 1, "*": 1, "/": 2,
            "square": 1, "sqrt": 2, "abs": 1,
            "log": 3, "exp": 3, "cube": 2,
            "sin": 3, "cos": 3,
        }
        
        # Build rationale
        rationale_parts = [
            f"Based on analysis of {len(features)} features:",
            f"- Data has {'negative' if has_negative else 'non-negative'} values",
            f"- Skewness: {'high' if is_skewed else 'normal'} (mean |skew|={skewness:.2f})",
            f"- Division: {'enabled' if '/' in binary_operators else 'disabled (near-zero values)'}",
            f"- Log transform: {'enabled' if 'log' in unary_operators else 'disabled'}",
        ]
        
        return SRPrimitives(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            constraints=constraints,
            complexity_weights={k: v for k, v in complexity_weights.items() 
                              if k in binary_operators + unary_operators},
            rationale="\n".join(rationale_parts),
        )
    
    def prepare_all_binary_datasets(
        self,
        df: pd.DataFrame,
        features: List[str],
        reference_class: str = "NORM",
        pathologies: Optional[List[str]] = None,
        output_dir: str = "data/sr_ready",
    ) -> Dict[str, SRDataset]:
        """
        Prepare binary classification datasets for all pathologies vs reference.
        
        Args:
            df: DataFrame with features
            features: Features to include
            reference_class: Reference class (usually 'NORM')
            pathologies: List of pathologies (None for all)
            output_dir: Directory to save datasets
            
        Returns:
            Dictionary mapping pathology name to SRDataset
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if pathologies is None:
            pathologies = ["MI", "HYP", "CD", "STTC", "OTHER"]
        
        datasets = {}
        
        for pathology in pathologies:
            dataset = self.prepare_binary_classification(
                df,
                positive_class=pathology,
                negative_class=reference_class,
                features=features,
                normalize=True,
                balance_classes=True,
            )
            
            # Save dataset
            filename = f"{reference_class.lower()}_vs_{pathology.lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            dataset.save(filepath)
            
            datasets[pathology] = dataset
        
        return datasets
    
    def generate_sr_config(
        self,
        dataset: SRDataset,
        primitives: SRPrimitives,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete PySR configuration.
        
        Args:
            dataset: SRDataset to use
            primitives: SRPrimitives with operators
            output_path: Optional path to save config as JSON
            
        Returns:
            Dictionary with PySR configuration
        """
        config = {
            "model_config": {
                "niterations": 100,
                "populations": 30,
                "population_size": 50,
                "ncycles_per_iteration": 300,
                "binary_operators": primitives.binary_operators,
                "unary_operators": primitives.unary_operators,
                "complexity_of_operators": primitives.complexity_weights,
                "maxsize": primitives.constraints.get("max_size", 30),
                "maxdepth": primitives.constraints.get("max_depth", 8),
                "parsimony": 0.0032,  # Complexity penalty
                "weight_optimize": 0.001,
                "turbo": True,
                "bumper": True,
            },
            "dataset_info": {
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "feature_names": dataset.feature_names,
                "target_name": dataset.target_name,
                "task_type": dataset.task_type,
            },
            "metadata": dataset.metadata,
            "primitives_rationale": primitives.rationale,
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def _balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes by undersampling the majority class."""
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        balanced_indices = []
        for c in classes:
            class_indices = np.where(y == c)[0]
            if len(class_indices) > min_count:
                np.random.seed(self.random_state)
                selected = np.random.choice(class_indices, min_count, replace=False)
            else:
                selected = class_indices
            balanced_indices.extend(selected)
        
        np.random.seed(self.random_state)
        np.random.shuffle(balanced_indices)
        
        return X[balanced_indices], y[balanced_indices]
    
    def split_dataset(
        self,
        dataset: SRDataset,
        test_size: Optional[float] = None,
    ) -> Tuple[SRDataset, SRDataset]:
        """
        Split dataset into train and test sets.
        
        Args:
            dataset: SRDataset to split
            test_size: Fraction for test set (default from init)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if test_size is None:
            test_size = self.test_size
        
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=dataset.y if dataset.task_type != "regression" else None,
        )
        
        train_dataset = SRDataset(
            X=X_train,
            y=y_train,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            n_samples=len(X_train),
            n_features=dataset.n_features,
            task_type=dataset.task_type,
            metadata={**dataset.metadata, "split": "train"},
        )
        
        test_dataset = SRDataset(
            X=X_test,
            y=y_test,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            n_samples=len(X_test),
            n_features=dataset.n_features,
            task_type=dataset.task_type,
            metadata={**dataset.metadata, "split": "test"},
        )
        
        return train_dataset, test_dataset
    
    def get_baseline_performance(
        self,
        dataset: SRDataset,
    ) -> Dict[str, float]:
        """
        Calculate baseline performance metrics for a dataset.
        
        Useful for comparing against symbolic regression results.
        
        Args:
            dataset: SRDataset to evaluate
            
        Returns:
            Dictionary with baseline metrics
        """
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
        
        # Split data
        train, test = self.split_dataset(dataset)
        
        if dataset.task_type in ["binary", "multiclass"]:
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            model.fit(train.X, train.y)
            
            y_pred = model.predict(test.X)
            y_proba = model.predict_proba(test.X)
            
            metrics = {
                "accuracy": accuracy_score(test.y, y_pred),
                "baseline_model": "LogisticRegression",
            }
            
            if dataset.task_type == "binary":
                metrics["roc_auc"] = roc_auc_score(test.y, y_proba[:, 1])
        
        else:  # regression
            model = LinearRegression()
            model.fit(train.X, train.y)
            
            y_pred = model.predict(test.X)
            
            metrics = {
                "r2_score": r2_score(test.y, y_pred),
                "rmse": np.sqrt(mean_squared_error(test.y, y_pred)),
                "baseline_model": "LinearRegression",
            }
        
        return metrics
