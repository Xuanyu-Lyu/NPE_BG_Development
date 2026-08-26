"""
Combine Neural Network Training Data from All Conditions - Large Dataset Version

This script combines the individual condition results from generate_training_data_cluster.py
into a single training dataset for neural network model training.

Usage:
    python combine_training_data.py [--split] [--test_size 0.2]
    
Options:
    --split: Create train/test split
    --test_size: Proportion of data for testing (default: 0.2)
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data_Vali/DataGeneratingNN_uniSEMPGS_20250223")

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_all_conditions():
    """Load all condition results and combine into a single dataset."""
    print("\n" + "="*70)
    print("COMBINING NEURAL NETWORK TRAINING DATA")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}\n")
    
    # Find all condition directories
    condition_dirs = sorted([d for d in PROJECT_BASE.iterdir() 
                            if d.is_dir() and d.name.startswith('Condition_')])
    
    print(f"Found {len(condition_dirs)} condition directories\n")
    
    all_data = []
    condition_summary = []
    
    for i, condition_dir in enumerate(condition_dirs, 1):
        condition_name = condition_dir.name
        
        # Load NN training format file
        nn_file = condition_dir / "nn_training_format.csv"
        
        if not nn_file.exists():
            print(f"{i:3d}. {condition_name}: ✗ No training file")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Missing',
                'N_Iterations': 0
            })
            continue
        
        try:
            df = pd.read_csv(nn_file)
            n_iterations = len(df)
            all_data.append(df)
            
            print(f"{i:3d}. {condition_name}: ✓ {n_iterations} iterations")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Success',
                'N_Iterations': n_iterations
            })
            
        except Exception as e:
            print(f"{i:3d}. {condition_name}: ✗ Error - {e}")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Error',
                'N_Iterations': 0
            })
    
    # Save condition summary
    summary_df = pd.DataFrame(condition_summary)
    summary_file = PROJECT_BASE / "data_collection_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Saved collection summary to {summary_file}")
    
    # Statistics
    n_success = sum(1 for s in condition_summary if s['Status'] == 'Success')
    n_total_iterations = sum(s['N_Iterations'] for s in condition_summary)
    
    print(f"\nSummary:")
    print(f"  Conditions with data: {n_success}/{len(condition_dirs)}")
    print(f"  Total iterations: {n_total_iterations}")
    
    if len(all_data) == 0:
        print("\n✗ No data to combine!")
        return None
    
    # Combine all data
    print(f"\nCombining data from {len(all_data)} conditions...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"✓ Combined dataset shape: {combined_df.shape}")
    print(f"  Rows: {len(combined_df)}")
    print(f"  Columns: {len(combined_df.columns)}")
    
    return combined_df

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train/test split at the condition level."""
    print(f"\nCreating train/test split (test_size={test_size})...")
    
    # Get unique conditions
    conditions = df['Condition'].unique()
    print(f"  Total conditions: {len(conditions)}")
    
    # Split conditions using numpy
    np.random.seed(random_state)
    n_test = int(len(conditions) * test_size)
    n_train = len(conditions) - n_test
    
    # Shuffle and split
    shuffled_conditions = np.random.permutation(conditions)
    test_conditions = shuffled_conditions[:n_test]
    train_conditions = shuffled_conditions[n_test:]
    
    print(f"  Train conditions: {len(train_conditions)}")
    print(f"  Test conditions: {len(test_conditions)}")
    
    # Create train/test datasets
    train_df = df[df['Condition'].isin(train_conditions)].copy()
    test_df = df[df['Condition'].isin(test_conditions)].copy()
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return train_df, test_df

def save_datasets(combined_df, train_df=None, test_df=None):
    """Save combined and split datasets."""
    print("\nSaving datasets...")
    
    # Save combined dataset
    combined_file = PROJECT_BASE / "nn_training_combined.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Saved combined dataset to {combined_file}")
    print(f"  Shape: {combined_df.shape}")
    
    # Save train/test splits if provided
    if train_df is not None and test_df is not None:
        train_file = PROJECT_BASE / "nn_training_train.csv"
        test_file = PROJECT_BASE / "nn_training_test.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"✓ Saved train dataset to {train_file}")
        print(f"  Shape: {train_df.shape}")
        print(f"✓ Saved test dataset to {test_file}")
        print(f"  Shape: {test_df.shape}")
    
    # Print column information
    print(f"\nDataset columns:")
    print(f"  Parameter columns: {[col for col in combined_df.columns if col.startswith('param_')]}")
    print(f"  Total correlation columns: {len([col for col in combined_df.columns if not col.startswith('param_') and col not in ['Iteration', 'Condition']])}")

def print_dataset_statistics(df):
    """Print statistics about the dataset."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Parameter ranges
    print("\nParameter ranges:")
    param_cols = [col for col in df.columns if col.startswith('param_')]
    for param_col in sorted(param_cols):
        values = df[param_col].dropna().unique()
        print(f"  {param_col:20s}: {len(values):2d} values - [{values.min():.3f}, {values.max():.3f}]")
    
    # Correlation columns
    corr_cols = [col for col in df.columns 
                 if not col.startswith('param_') 
                 and col not in ['Iteration', 'Condition']
                 and not col.endswith('_N')]
    
    print(f"\nCorrelation features: {len(corr_cols)}")
    
    # Check for missing data
    missing_counts = df[corr_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print("\nColumns with missing data:")
        for col in missing_counts[missing_counts > 0].index:
            pct = 100 * missing_counts[col] / len(df)
            print(f"  {col}: {missing_counts[col]} ({pct:.1f}%)")
    else:
        print("\n✓ No missing data in correlation columns")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Combine NN training data from all conditions')
    parser.add_argument('--split', action='store_true', 
                       help='Create train/test split')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    args = parser.parse_args()
    
    # Load and combine all condition data
    combined_df = load_all_conditions()
    
    if combined_df is None:
        return
    
    # Print statistics
    print_dataset_statistics(combined_df)
    
    # Create train/test split if requested
    train_df = None
    test_df = None
    
    if args.split:
        train_df, test_df = create_train_test_split(combined_df, test_size=args.test_size)
    
    # Save datasets
    save_datasets(combined_df, train_df, test_df)
    
    print("\n" + "="*70)
    print("DATA COMBINATION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
