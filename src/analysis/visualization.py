"""
ECG Visualization Module for Kepler-ECG Phase 3.

This module provides comprehensive visualization tools for exploring
ECG features across diagnostic categories, including:
- Box plots and violin plots for feature distributions
- Correlation heatmaps
- Dimensionality reduction (t-SNE, PCA)
- Compressibility landscape plots

All plots use a consistent style optimized for publication.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Define consistent color palette for diagnostic categories
DIAGNOSIS_COLORS = {
    "NORM": "#2ecc71",   # Green - healthy
    "MI": "#e74c3c",     # Red - myocardial infarction
    "HYP": "#9b59b6",    # Purple - hypertrophy
    "CD": "#3498db",     # Blue - conduction disturbance
    "STTC": "#f39c12",   # Orange - ST/T changes
    "OTHER": "#1abc9c",  # Teal - other conditions
    "UNKNOWN": "#95a5a6", # Gray
}

# Diagnosis order for consistent plotting
DIAGNOSIS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP", "OTHER"]


class ECGVisualizer:
    """
    Visualization tools for ECG feature analysis.
    
    Provides methods for creating publication-quality plots to explore
    relationships between ECG features and diagnostic categories.
    
    Example:
        >>> viz = ECGVisualizer()
        >>> fig = viz.plot_feature_by_category(df, 'comp_sig_gzip_ratio')
        >>> fig.savefig('gzip_by_category.png')
    """
    
    def __init__(
        self,
        style: str = "whitegrid",
        context: str = "paper",
        palette: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
    ):
        """
        Initialize the visualizer with style settings.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
            palette: Custom color palette for categories
            figsize: Default figure size (width, height)
            dpi: Figure resolution
        """
        self.style = style
        self.context = context
        self.palette = palette or DIAGNOSIS_COLORS
        self.figsize = figsize
        self.dpi = dpi
        
        # Apply style settings
        sns.set_style(style)
        sns.set_context(context)
    
    def plot_feature_by_category(
        self,
        df: pd.DataFrame,
        feature: str,
        category_col: str = "diag_primary_category",
        plot_type: str = "boxplot",
        exclude_categories: Optional[List[str]] = None,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_points: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """
        Create boxplot or violin plot of a feature by diagnostic category.
        
        Args:
            df: DataFrame with feature and category columns
            feature: Name of feature column
            category_col: Name of category column
            plot_type: 'boxplot', 'violin', or 'box_swarm'
            exclude_categories: Categories to exclude (default: ['UNKNOWN'])
            title: Plot title (auto-generated if None)
            ylabel: Y-axis label (uses feature name if None)
            show_points: Overlay individual points on boxplot
            figsize: Figure size override
            
        Returns:
            Matplotlib Figure object
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Filter data
        data = df[[feature, category_col]].dropna()
        data = data[~data[category_col].isin(exclude_categories)]
        
        # Order categories
        order = [c for c in DIAGNOSIS_ORDER if c in data[category_col].unique()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Get colors for categories in order
        colors = [self.palette.get(c, "#333333") for c in order]
        
        if plot_type == "boxplot":
            sns.boxplot(
                data=data,
                x=category_col,
                y=feature,
                order=order,
                palette=colors,
                ax=ax,
                width=0.6,
            )
            if show_points:
                sns.stripplot(
                    data=data,
                    x=category_col,
                    y=feature,
                    order=order,
                    color="black",
                    alpha=0.3,
                    size=2,
                    ax=ax,
                )
        
        elif plot_type == "violin":
            sns.violinplot(
                data=data,
                x=category_col,
                y=feature,
                order=order,
                palette=colors,
                ax=ax,
                inner="box",
                cut=0,
            )
        
        elif plot_type == "box_swarm":
            sns.boxplot(
                data=data,
                x=category_col,
                y=feature,
                order=order,
                palette=colors,
                ax=ax,
                width=0.5,
                fliersize=0,
            )
            sns.swarmplot(
                data=data.sample(min(500, len(data))),  # Limit points for speed
                x=category_col,
                y=feature,
                order=order,
                color="black",
                alpha=0.5,
                size=2,
                ax=ax,
            )
        
        # Labels and title
        ax.set_xlabel("Diagnostic Category", fontsize=12)
        ax.set_ylabel(ylabel or feature, fontsize=12)
        ax.set_title(title or f"{feature} by Diagnostic Category", fontsize=14)
        
        # Add sample sizes
        for i, cat in enumerate(order):
            n = len(data[data[category_col] == cat])
            ax.annotate(
                f"n={n}",
                xy=(i, ax.get_ylim()[0]),
                ha="center",
                va="top",
                fontsize=9,
                color="gray",
            )
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        feature_prefix: Optional[str] = None,
        method: str = "pearson",
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        annot: bool = True,
        mask_upper: bool = True,
    ) -> Figure:
        """
        Create correlation heatmap for a set of features.
        
        Args:
            df: DataFrame with features
            features: List of feature names
            feature_prefix: Alternative: filter features by prefix
            method: Correlation method ('pearson', 'spearman')
            title: Plot title
            figsize: Figure size
            annot: Show correlation values
            mask_upper: Mask upper triangle
            
        Returns:
            Matplotlib Figure object
        """
        if features is None and feature_prefix is None:
            raise ValueError("Must provide either features or feature_prefix")
        
        if features is None:
            features = [c for c in df.columns if c.startswith(feature_prefix)]
        
        # Calculate correlation matrix
        corr = df[features].corr(method=method)
        
        # Create mask for upper triangle
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        # Determine figure size based on number of features
        if figsize is None:
            size = max(8, len(features) * 0.6)
            figsize = (size, size * 0.8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Shorten feature names for readability
        short_names = [f.replace(feature_prefix or "", "").replace("comp_", "").replace("wav_", "").replace("hrv_", "") 
                       for f in features]
        corr_display = corr.copy()
        corr_display.index = short_names
        corr_display.columns = short_names
        
        sns.heatmap(
            corr_display,
            mask=mask,
            annot=annot,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
            ax=ax,
            annot_kws={"size": 8},
        )
        
        ax.set_title(title or f"Feature Correlation Matrix ({method.capitalize()})", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str],
        hue: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
        ncols: int = 3,
        figsize_per_plot: Tuple[float, float] = (4, 3),
        title: Optional[str] = None,
    ) -> Figure:
        """
        Create grid of distribution plots for multiple features.
        
        Args:
            df: DataFrame with features
            features: List of feature names to plot
            hue: Column to use for coloring (category)
            exclude_categories: Categories to exclude
            ncols: Number of columns in grid
            figsize_per_plot: Size of each subplot
            title: Overall figure title
            
        Returns:
            Matplotlib Figure object
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        data = df.copy()
        data = data[~data[hue].isin(exclude_categories)]
        
        # Calculate grid dimensions
        nrows = (len(features) + ncols - 1) // ncols
        figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        axes = np.atleast_2d(axes).flatten()
        
        # Order for legend
        order = [c for c in DIAGNOSIS_ORDER if c in data[hue].unique()]
        colors = [self.palette.get(c, "#333333") for c in order]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            for cat, color in zip(order, colors):
                subset = data[data[hue] == cat][feature].dropna()
                if len(subset) > 0:
                    sns.kdeplot(
                        data=subset,
                        ax=ax,
                        color=color,
                        label=cat,
                        fill=True,
                        alpha=0.3,
                    )
            
            ax.set_xlabel(feature.split("_")[-1], fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(feature, fontsize=10)
            
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.get_legend() and ax.get_legend().remove()
        
        # Hide empty subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_dimensionality_reduction(
        self,
        df: pd.DataFrame,
        features: List[str],
        method: str = "tsne",
        hue: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sample_size: Optional[int] = 5000,
    ) -> Figure:
        """
        Create t-SNE or PCA visualization colored by category.
        
        Args:
            df: DataFrame with features
            features: List of feature names to use
            method: 'tsne' or 'pca'
            hue: Column for coloring points
            exclude_categories: Categories to exclude
            n_components: Number of components (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility
            title: Plot title
            figsize: Figure size
            sample_size: Max samples to plot (for speed)
            
        Returns:
            Matplotlib Figure object
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Prepare data
        data = df[features + [hue]].dropna()
        data = data[~data[hue].isin(exclude_categories)]
        
        # Sample if too large
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=random_state)
        
        X = data[features].values
        y = data[hue].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
            )
            method_name = "t-SNE"
        elif method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            method_name = "PCA"
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_reduced = reducer.fit_transform(X_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or (10, 8), dpi=self.dpi)
        
        # Plot each category
        order = [c for c in DIAGNOSIS_ORDER if c in np.unique(y)]
        
        for cat in order:
            mask = y == cat
            ax.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=self.palette.get(cat, "#333333"),
                label=f"{cat} (n={mask.sum()})",
                alpha=0.6,
                s=20,
                edgecolors="white",
                linewidths=0.3,
            )
        
        ax.set_xlabel(f"{method_name} Component 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Component 2", fontsize=12)
        ax.set_title(
            title or f"{method_name} Projection of ECG Features by Diagnosis",
            fontsize=14,
        )
        ax.legend(loc="best", fontsize=10)
        
        # Add variance explained for PCA
        if method.lower() == "pca":
            var_exp = reducer.explained_variance_ratio_
            ax.annotate(
                f"Variance explained: {var_exp[0]:.1%} + {var_exp[1]:.1%} = {sum(var_exp[:2]):.1%}",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                fontsize=10,
                va="top",
            )
        
        plt.tight_layout()
        return fig
    
    def plot_compressibility_landscape(
        self,
        df: pd.DataFrame,
        x_feature: str = "comp_sig_gzip_ratio",
        y_feature: str = "comp_rr_approx_entropy",
        hue: str = "diag_primary_category",
        exclude_categories: Optional[List[str]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        sample_size: Optional[int] = 5000,
        show_density: bool = True,
    ) -> Figure:
        """
        Create 2D 'landscape' visualization of compressibility metrics.
        
        Args:
            df: DataFrame with features
            x_feature: Feature for X axis
            y_feature: Feature for Y axis
            hue: Column for coloring
            exclude_categories: Categories to exclude
            title: Plot title
            figsize: Figure size
            sample_size: Max samples to plot
            show_density: Show KDE contours
            
        Returns:
            Matplotlib Figure object
        """
        if exclude_categories is None:
            exclude_categories = ["UNKNOWN"]
        
        # Prepare data
        data = df[[x_feature, y_feature, hue]].dropna()
        data = data[~data[hue].isin(exclude_categories)]
        
        # Sample if needed
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        fig, ax = plt.subplots(figsize=figsize or (12, 8), dpi=self.dpi)
        
        order = [c for c in DIAGNOSIS_ORDER if c in data[hue].unique()]
        
        # Plot scatter with optional density
        for cat in order:
            subset = data[data[hue] == cat]
            color = self.palette.get(cat, "#333333")
            
            ax.scatter(
                subset[x_feature],
                subset[y_feature],
                c=color,
                label=f"{cat} (n={len(subset)})",
                alpha=0.4,
                s=15,
                edgecolors="none",
            )
            
            if show_density and len(subset) > 50:
                try:
                    sns.kdeplot(
                        data=subset,
                        x=x_feature,
                        y=y_feature,
                        color=color,
                        levels=3,
                        linewidths=1.5,
                        ax=ax,
                    )
                except Exception:
                    pass  # Skip if KDE fails
        
        ax.set_xlabel(x_feature, fontsize=12)
        ax.set_ylabel(y_feature, fontsize=12)
        ax.set_title(
            title or "Compressibility Landscape by Diagnostic Category",
            fontsize=14,
        )
        ax.legend(loc="best", fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_norm_vs_pathological(
        self,
        df: pd.DataFrame,
        features: List[str],
        category_col: str = "diag_primary_category",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """
        Create violin plots comparing NORM vs all pathological categories.
        
        Args:
            df: DataFrame with features
            features: List of features to plot
            category_col: Category column name
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        # Create binary classification
        data = df.copy()
        data = data[data[category_col] != "UNKNOWN"]
        data["pathological"] = data[category_col].apply(
            lambda x: "Normal" if x == "NORM" else "Pathological"
        )
        
        nrows = (len(features) + 2) // 3
        ncols = min(3, len(features))
        
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        axes = np.atleast_2d(axes).flatten()
        
        colors = {"Normal": self.palette["NORM"], "Pathological": "#e74c3c"}
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            plot_data = data[[feature, "pathological"]].dropna()
            
            sns.violinplot(
                data=plot_data,
                x="pathological",
                y=feature,
                order=["Normal", "Pathological"],
                palette=colors,
                ax=ax,
                inner="box",
                cut=0,
            )
            
            ax.set_xlabel("")
            ax.set_ylabel(feature, fontsize=10)
            ax.set_title(feature.split("_")[-2:], fontsize=10)
            
            # Add sample sizes
            for i, label in enumerate(["Normal", "Pathological"]):
                n = len(plot_data[plot_data["pathological"] == label])
                ax.annotate(
                    f"n={n}",
                    xy=(i, ax.get_ylim()[0]),
                    ha="center",
                    fontsize=9,
                    color="gray",
                )
        
        # Hide empty subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle("Normal vs Pathological ECG Features", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_bar(
        self,
        importance_df: pd.DataFrame,
        n_features: int = 15,
        metric: str = "f_statistic",
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """
        Create horizontal bar chart of feature importance.
        
        Args:
            importance_df: DataFrame with feature importance (from StatisticalAnalyzer)
            n_features: Number of top features to show
            metric: Column to use for ranking ('f_statistic', 'max_cohens_d')
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        # Get top features
        top_features = importance_df.nsmallest(n_features, "p_value")
        
        fig, ax = plt.subplots(figsize=figsize or (10, 8), dpi=self.dpi)
        
        # Create bar chart
        y_pos = np.arange(len(top_features))
        values = top_features[metric].values
        
        # Color by effect size
        colors = []
        for d in top_features["max_cohens_d"]:
            if d >= 0.8:
                colors.append("#27ae60")  # Large - green
            elif d >= 0.5:
                colors.append("#3498db")  # Medium - blue
            elif d >= 0.2:
                colors.append("#f39c12")  # Small - orange
            else:
                colors.append("#95a5a6")  # Negligible - gray
        
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        ax.set_title(title or f"Top {n_features} Discriminative Features", fontsize=14)
        
        # Add p-value annotations
        for i, (idx, row) in enumerate(top_features.iterrows()):
            p_str = f"p={row['p_value']:.1e}" if row['p_value'] > 0 else "p<1e-300"
            ax.annotate(
                p_str,
                xy=(values[i], i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="gray",
            )
        
        # Legend for effect sizes
        legend_elements = [
            mpatches.Patch(color="#27ae60", label="Large (d≥0.8)"),
            mpatches.Patch(color="#3498db", label="Medium (d≥0.5)"),
            mpatches.Patch(color="#f39c12", label="Small (d≥0.2)"),
            mpatches.Patch(color="#95a5a6", label="Negligible (d<0.2)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, title="Effect Size")
        
        plt.tight_layout()
        return fig
    
    def save_figure(
        self,
        fig: Figure,
        filename: str,
        formats: List[str] = ["png", "pdf"],
        output_dir: str = "reports",
    ) -> List[str]:
        """
        Save figure in multiple formats.
        
        Args:
            fig: Matplotlib Figure object
            filename: Base filename (without extension)
            formats: List of formats to save
            output_dir: Output directory
            
        Returns:
            List of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for fmt in formats:
            path = os.path.join(output_dir, f"{filename}.{fmt}")
            fig.savefig(path, format=fmt, bbox_inches="tight", dpi=self.dpi)
            saved_paths.append(path)
        
        return saved_paths
