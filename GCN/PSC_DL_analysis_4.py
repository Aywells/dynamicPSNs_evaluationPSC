"""
Statistical Comparison of Classification Methods
Performs pairwise significance testing and p-value correction
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_results(filepath):
    """Load the results Excel file"""
    df = pd.read_excel(filepath, sheet_name='results_PSC_DL')
    return df


def extract_methods(df, metric_suffix):
    """
    Extract method names based on metric suffix
    
    Parameters:
    -----------
    df : pandas DataFrame
        The results dataframe
    metric_suffix : str
        Either '_agg_misclass' or '_run_time'
    
    Returns:
    --------
    list of method names (without suffix)
    """
    method_cols = [col for col in df.columns if col.endswith(metric_suffix)]
    methods = [col.replace(metric_suffix, '') for col in method_cols]
    return methods, method_cols


def paired_t_test(method1_data, method2_data):
    """
    Perform paired t-test between two methods
    
    Returns:
    --------
    dict with 't_statistic', 'p_value', and 'mean_difference'
    """
    # Remove any NaN pairs
    valid_idx = ~(pd.isna(method1_data) | pd.isna(method2_data))
    m1 = method1_data[valid_idx]
    m2 = method2_data[valid_idx]
    
    if len(m1) < 2:
        return {'t_statistic': np.nan, 'p_value': np.nan, 'mean_difference': np.nan}
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(m1, m2)
    mean_diff = np.mean(m1 - m2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff
    }


def wilcoxon_test(method1_data, method2_data):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative)
    
    Returns:
    --------
    dict with 'statistic', 'p_value'
    """
    # Remove any NaN pairs
    valid_idx = ~(pd.isna(method1_data) | pd.isna(method2_data))
    m1 = method1_data[valid_idx]
    m2 = method2_data[valid_idx]
    
    if len(m1) < 2:
        return {'statistic': np.nan, 'p_value': np.nan}
    
    try:
        stat, p_value = stats.wilcoxon(m1, m2)
        return {'statistic': stat, 'p_value': p_value}
    except:
        return {'statistic': np.nan, 'p_value': np.nan}


def pairwise_comparison(df, metric_suffix, test_type='t-test'):
    """
    Perform all pairwise comparisons between methods
    
    Parameters:
    -----------
    df : pandas DataFrame
        The results dataframe
    metric_suffix : str
        Either '_agg_misclass' or '_run_time'
    test_type : str
        Either 't-test' or 'wilcoxon'
    
    Returns:
    --------
    DataFrame with pairwise comparison results
    """
    methods, method_cols = extract_methods(df, metric_suffix)
    
    results = []
    
    # Perform all pairwise comparisons
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i != j:  # Compare all pairs (including reverse)
                col1 = method_cols[i]
                col2 = method_cols[j]
                
                if test_type == 't-test':
                    test_result = paired_t_test(df[col1], df[col2])
                    results.append({
                        'method_1': method1,
                        'method_2': method2,
                        'mean_1': df[col1].mean(),
                        'mean_2': df[col2].mean(),
                        'mean_difference': test_result['mean_difference'],
                        't_statistic': test_result['t_statistic'],
                        'p_value': test_result['p_value']
                    })
                else:  # wilcoxon
                    test_result = wilcoxon_test(df[col1], df[col2])
                    results.append({
                        'method_1': method1,
                        'method_2': method2,
                        'median_1': df[col1].median(),
                        'median_2': df[col2].median(),
                        'statistic': test_result['statistic'],
                        'p_value': test_result['p_value']
                    })
    
    return pd.DataFrame(results)


def apply_corrections(pairwise_results):
    """
    Apply multiple testing corrections to p-values
    
    Returns:
    --------
    DataFrame with corrected p-values
    """
    results_copy = pairwise_results.copy()
    
    # Extract valid p-values
    valid_pvals = results_copy['p_value'].dropna()
    
    if len(valid_pvals) == 0:
        results_copy['p_value_bonferroni'] = np.nan
        results_copy['p_value_fdr_bh'] = np.nan
        return results_copy
    
    # Bonferroni correction
    bonferroni_corrected = np.minimum(valid_pvals * len(valid_pvals), 1.0)
    
    # Benjamini-Hochberg FDR correction
    fdr_corrected = np.zeros_like(valid_pvals)
    sorted_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals.iloc[sorted_idx].values
    
    n = len(sorted_pvals)
    for i, (idx, pval) in enumerate(zip(sorted_idx, sorted_pvals)):
        fdr_corrected[idx] = min(pval * n / (i + 1), 1.0)
    
    # Enforce monotonicity for FDR
    for i in range(n-2, -1, -1):
        sorted_rev_idx = sorted_idx[::-1]
        fdr_corrected[sorted_rev_idx[i]] = min(
            fdr_corrected[sorted_rev_idx[i]], 
            fdr_corrected[sorted_rev_idx[i+1]]
        )
    
    # Add corrected p-values to dataframe
    results_copy.loc[valid_pvals.index, 'p_value_bonferroni'] = bonferroni_corrected.values
    results_copy.loc[valid_pvals.index, 'p_value_fdr_bh'] = fdr_corrected
    
    return results_copy


def create_pvalue_matrix(pairwise_results, methods, correction=None):
    """
    Create a matrix of p-values for visualization
    
    Parameters:
    -----------
    pairwise_results : DataFrame
        Results from pairwise_comparison
    methods : list
        List of method names
    correction : str or None
        'bonferroni', 'fdr_bh', or None for uncorrected
    
    Returns:
    --------
    DataFrame matrix with methods as rows and columns
    """
    pval_col = 'p_value'
    if correction == 'bonferroni':
        pval_col = 'p_value_bonferroni'
    elif correction == 'fdr_bh':
        pval_col = 'p_value_fdr_bh'
    
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    
    for _, row in pairwise_results.iterrows():
        m1 = row['method_1']
        m2 = row['method_2']
        matrix.loc[m1, m2] = row[pval_col]
    
    # Diagonal is always 1.0 (comparing to self)
    for method in methods:
        matrix.loc[method, method] = 1.0
    
    return matrix


def main():
    """Main execution function"""
    
    # Load data
    print("Loading data...")
    filepath = 'results_PSC_DL.xlsx'  # Update this path as needed
    df = pd.read_excel(filepath)
    
    print(f"Loaded {len(df)} datasets\n")
    
    # ==========================================
    # MISCLASSIFICATION RATE COMPARISONS
    # ==========================================
    print("="*60)
    print("MISCLASSIFICATION RATE COMPARISONS")
    print("="*60)
    
    methods_misclass, _ = extract_methods(df, '_agg_misclass')
    print(f"\nFound {len(methods_misclass)} methods:")
    for i, method in enumerate(methods_misclass, 1):
        print(f"  {i}. {method}")
    
    # Paired t-test
    print("\n\nPerforming paired t-tests...")
    misclass_results = pairwise_comparison(df, '_agg_misclass', test_type='t-test')
    
    # Apply corrections
    print("Applying multiple testing corrections...")
    misclass_results = apply_corrections(misclass_results)
    
    # Create p-value matrices
    print("Creating p-value matrices...")
    pval_matrix_raw = create_pvalue_matrix(misclass_results, methods_misclass, correction=None)
    pval_matrix_bonf = create_pvalue_matrix(misclass_results, methods_misclass, correction='bonferroni')
    pval_matrix_fdr = create_pvalue_matrix(misclass_results, methods_misclass, correction='fdr_bh')
    
    # ==========================================
    # RUNTIME COMPARISONS
    # ==========================================
    print("\n" + "="*60)
    print("RUNTIME COMPARISONS")
    print("="*60)
    
    methods_runtime, _ = extract_methods(df, '_run_time')
    print(f"\nFound {len(methods_runtime)} methods:")
    for i, method in enumerate(methods_runtime, 1):
        print(f"  {i}. {method}")
    
    # Paired t-test
    print("\n\nPerforming paired t-tests...")
    runtime_results = pairwise_comparison(df, '_run_time', test_type='t-test')
    
    # Apply corrections
    print("Applying multiple testing corrections...")
    runtime_results = apply_corrections(runtime_results)
    
    # Create p-value matrices
    print("Creating p-value matrices...")
    runtime_pval_matrix_raw = create_pvalue_matrix(runtime_results, methods_runtime, correction=None)
    runtime_pval_matrix_bonf = create_pvalue_matrix(runtime_results, methods_runtime, correction='bonferroni')
    runtime_pval_matrix_fdr = create_pvalue_matrix(runtime_results, methods_runtime, correction='fdr_bh')
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_file = 'statistical_comparison_results.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Misclassification results
        misclass_results.to_excel(writer, sheet_name='Misclass_Pairwise', index=False)
        pval_matrix_raw.to_excel(writer, sheet_name='Misclass_PValues_Raw')
        pval_matrix_bonf.to_excel(writer, sheet_name='Misclass_PValues_Bonf')
        pval_matrix_fdr.to_excel(writer, sheet_name='Misclass_PValues_FDR')
        
        # Runtime results
        runtime_results.to_excel(writer, sheet_name='Runtime_Pairwise', index=False)
        runtime_pval_matrix_raw.to_excel(writer, sheet_name='Runtime_PValues_Raw')
        runtime_pval_matrix_bonf.to_excel(writer, sheet_name='Runtime_PValues_Bonf')
        runtime_pval_matrix_fdr.to_excel(writer, sheet_name='Runtime_PValues_FDR')
    
    print(f"\nResults saved to: {output_file}")
    
    # ==========================================
    # SUMMARY STATISTICS
    # ==========================================
    print("\n" + "="*60)
    print("SUMMARY - SIGNIFICANT COMPARISONS (α=0.05)")
    print("="*60)
    
    # Misclassification
    sig_misclass_raw = (misclass_results['p_value'] < 0.05).sum()
    sig_misclass_bonf = (misclass_results['p_value_bonferroni'] < 0.05).sum()
    sig_misclass_fdr = (misclass_results['p_value_fdr_bh'] < 0.05).sum()
    
    print("\nMisclassification Rate:")
    print(f"  Total comparisons: {len(misclass_results)}")
    print(f"  Significant (uncorrected): {sig_misclass_raw}")
    print(f"  Significant (Bonferroni): {sig_misclass_bonf}")
    print(f"  Significant (FDR): {sig_misclass_fdr}")
    
    # Runtime
    sig_runtime_raw = (runtime_results['p_value'] < 0.05).sum()
    sig_runtime_bonf = (runtime_results['p_value_bonferroni'] < 0.05).sum()
    sig_runtime_fdr = (runtime_results['p_value_fdr_bh'] < 0.05).sum()
    
    print("\nRuntime:")
    print(f"  Total comparisons: {len(runtime_results)}")
    print(f"  Significant (uncorrected): {sig_runtime_raw}")
    print(f"  Significant (Bonferroni): {sig_runtime_bonf}")
    print(f"  Significant (FDR): {sig_runtime_fdr}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()