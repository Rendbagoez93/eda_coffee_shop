"""Base visualizer with common plotting utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger


class BaseVisualizer:
    """Base class for all visualizers with common plotting utilities."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize BaseVisualizer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        # Set up visualization style
        self.setup_style()
        
        # Output directory
        self.output_dir = Path(self.paths.get('output', {}).get('root', 'output'))
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_style(self):
        """Set up matplotlib and seaborn style."""
        # Get style from config or use defaults
        viz_config = self.config.get('visualization', {})
        
        style = viz_config.get('style', 'seaborn-v0_8-darkgrid')
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            sns.set_theme()
        
        # Set color palette
        palette = viz_config.get('color_palette', 'viridis')
        sns.set_palette(palette)
        
        # Set default figure size
        default_figsize = viz_config.get('figure_size', {}).get('default', [12, 6])
        plt.rcParams['figure.figsize'] = default_figsize
        
        # Set DPI for high quality
        dpi = viz_config.get('dpi', 300)
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        
        # Font settings
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load CSV data from output directory.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with loaded data
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        return pd.read_csv(filepath)
    
    def load_json_data(self, filename: str) -> Dict:
        """
        Load JSON data from output directory.
        
        Args:
            filename: JSON filename
            
        Returns:
            Dictionary with loaded data
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_figure(
        self,
        fig,
        filename: str,
        tight: bool = True,
        close: bool = True
    ) -> Path:
        """
        Save figure to visualization directory.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            tight: Use tight layout
            close: Close figure after saving
            
        Returns:
            Path to saved file
        """
        filepath = self.viz_dir / filename
        
        if tight:
            fig.tight_layout()
        
        fig.savefig(filepath, bbox_inches='tight', dpi=self.config.get('visualization', {}).get('dpi', 300))
        
        if close:
            plt.close(fig)
        
        self.logger.info(f"Saved visualization: {filepath}")
        
        return filepath
    
    def format_currency(self, value: float) -> str:
        """Format value as currency."""
        return f"${value:,.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format value as percentage."""
        return f"{value:.1f}%"
    
    def add_value_labels(
        self,
        ax,
        orientation: str = 'vertical',
        format_func=None,
        size: int = 9
    ):
        """
        Add value labels to bar chart.
        
        Args:
            ax: Matplotlib axes
            orientation: 'vertical' or 'horizontal'
            format_func: Function to format values
            size: Font size
        """
        if format_func is None:
            format_func = lambda x: f'{x:.0f}'
        
        if orientation == 'vertical':
            for container in ax.containers:
                ax.bar_label(container, fmt=format_func, size=size)
        else:
            for container in ax.containers:
                ax.bar_label(container, fmt=format_func, size=size)
    
    def create_grid_layout(
        self,
        rows: int,
        cols: int,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create grid layout for multiple subplots.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size (width, height)
            
        Returns:
            Tuple of (figure, axes array)
        """
        if figsize is None:
            figsize = (cols * 6, rows * 5)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows * cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        return fig, axes
    
    def add_title_and_subtitle(
        self,
        ax,
        title: str,
        subtitle: Optional[str] = None
    ):
        """
        Add title and optional subtitle to axes.
        
        Args:
            ax: Matplotlib axes
            title: Main title
            subtitle: Optional subtitle
        """
        if subtitle:
            ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
