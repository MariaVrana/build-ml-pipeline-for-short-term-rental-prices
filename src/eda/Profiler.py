""" Profiler class that performs basic EDA (written using mostly copilot (Claude Sonnet 4) """
# This is not part of the exercise. 
# I could not find any profiling library compatible with python 3.13 so I used copilot to build a profiler class from scratch.

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Profiler:
    """
    A comprehensive EDA profiler class compatible with Python 3.13
    """
    
    def __init__(self, df, title="Dataset Profile"):
        self.df = df.copy()
        self.title = title
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def overview(self):
        """Display basic dataset information"""
        print(f"📊 {self.title}")
        print("=" * 50)
        print(f"📏 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"🔢 Numeric columns: {len(self.numeric_cols)}")
        print(f"📝 Categorical columns: {len(self.categorical_cols)}")
        print(f"📅 Datetime columns: {len(self.datetime_cols)}")
        print()
        
    def missing_values(self):
        """Analyze missing values"""
        print("🔍 Missing Values Analysis")
        print("-" * 30)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing %', ascending=False)
        
        # Display missing values table
        print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
        
        if missing_df['Missing Count'].sum() == 0:
            print("✅ No missing values found!")
        
        # Visualize missing values
        if missing.sum() > 0:
            plt.figure(figsize=(12, 6))
            
            # Missing values heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            
            # Missing values bar plot
            plt.subplot(1, 2, 2)
            missing_cols = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_cols) > 0:
                plt.barh(missing_cols['Column'], missing_cols['Missing %'])
                plt.xlabel('Missing Percentage')
                plt.title('Missing Values by Column')
            
            plt.tight_layout()
            plt.show()
        
        return missing_df
    
    def data_types_summary(self):
        """Summarize data types and unique values"""
        print("\n📋 Data Types Summary")
        print("-" * 25)
        
        dtype_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.values,
            'Unique Values': [self.df[col].nunique() for col in self.df.columns],
            'Sample Values': [str(self.df[col].dropna().iloc[:3].tolist()) for col in self.df.columns]
        })
        
        print(dtype_summary.to_string(index=False))
        return dtype_summary
    
    def numeric_analysis(self):
        """Analyze numeric columns"""
        if not self.numeric_cols:
            print("No numeric columns found.")
            return
            
        print(f"\n🔢 Numeric Columns Analysis ({len(self.numeric_cols)} columns)")
        print("-" * 40)
        
        # Statistical summary
        numeric_stats = self.df[self.numeric_cols].describe()
        print(numeric_stats)
        
        # Visualizations
        n_cols = min(4, len(self.numeric_cols))
        n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_cols):
            if i < len(axes):
                # Distribution plot
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(self.numeric_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        return numeric_stats
    
    def categorical_analysis(self):
        """Analyze categorical columns"""
        if not self.categorical_cols:
            print("No categorical columns found.")
            return
            
        print(f"\n📝 Categorical Columns Analysis ({len(self.categorical_cols)} columns)")
        print("-" * 45)
        
        cat_summary = []
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            most_frequent = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else "N/A"
            most_frequent_count = self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
            
            cat_summary.append({
                'Column': col,
                'Unique Values': unique_count,
                'Most Frequent': most_frequent,
                'Frequency': most_frequent_count
            })
        
        cat_df = pd.DataFrame(cat_summary)
        print(cat_df.to_string(index=False))
        
        # Visualize categorical distributions
        n_cols = min(3, len(self.categorical_cols))
        n_rows = (len(self.categorical_cols) + n_cols - 1) // n_cols
        
        if len(self.categorical_cols) > 0:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.categorical_cols):
                if i < len(axes):
                    top_categories = self.df[col].value_counts().head(10)
                    top_categories.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{col} Distribution (Top 10)')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Remove empty subplots
            for i in range(len(self.categorical_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()
        
        return cat_df
    
    def correlation_analysis(self):
        """Analyze correlations between numeric variables"""
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis.")
            return
            
        print(f"\n🔗 Correlation Analysis")
        print("-" * 25)
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Display correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            print("\n🚨 Highly Correlated Pairs (|correlation| > 0.7):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.to_string(index=False))
        else:
            print("\n✅ No highly correlated pairs found.")
        
        return corr_matrix
    
    def outliers_analysis(self):
        """Detect outliers in numeric columns"""
        if not self.numeric_cols:
            print("No numeric columns for outlier analysis.")
            return
            
        print(f"\n🎯 Outliers Analysis")
        print("-" * 20)
        
        outliers_summary = []
        
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_pct = (outliers_count / len(self.df)) * 100
            
            outliers_summary.append({
                'Column': col,
                'Outliers Count': outliers_count,
                'Outliers %': round(outliers_pct, 2),
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2)
            })
        
        outliers_df = pd.DataFrame(outliers_summary)
        print(outliers_df.to_string(index=False))
        
        # Box plots for outlier visualization
        if len(self.numeric_cols) > 0:
            n_cols = min(4, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(self.numeric_cols):
                if i < len(axes):
                    self.df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'{col} - Outliers Detection')
            
            # Remove empty subplots
            for i in range(len(self.numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()
        
        return outliers_df
    
    def generate_full_report(self):
        """Generate complete EDA report"""
        print("🚀 Generating Full EDA Report")
        print("=" * 50)
        
        # Run all analyses
        self.overview()
        missing_df = self.missing_values()
        dtype_df = self.data_types_summary()
        numeric_stats = self.numeric_analysis()
        cat_df = self.categorical_analysis()
        corr_matrix = self.correlation_analysis()
        outliers_df = self.outliers_analysis()
        
        print("\n✅ EDA Report Complete!")
        
        return {
            'missing_values': missing_df,
            'data_types': dtype_df,
            'numeric_stats': numeric_stats,
            'categorical_summary': cat_df,
            'correlation_matrix': corr_matrix,
            'outliers': outliers_df
        }

