"""Statistical analysis for reflection study."""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import warnings


def compute_ethics_accuracy(results):
    """
    Compute accuracy metrics for ETHICS.

    Returns DataFrame with accuracy, std, n, and extraction failure rate
    by level and thinking condition.
    """
    # Filter out error rows
    valid = results[results['extracted_answer'].notna() | results['correct'].notna()].copy()

    # Overall accuracy by condition
    accuracy = results.groupby(['level', 'thinking']).agg({
        'correct': ['mean', 'std', 'count'],
        'extracted_answer': lambda x: x.isna().mean()
    }).round(4)

    accuracy.columns = ['accuracy', 'std', 'n', 'extraction_failure']

    return accuracy


def compute_ethics_accuracy_by_subscale(results):
    """
    Compute accuracy metrics for ETHICS by subscale.

    Returns DataFrame with accuracy by level, thinking, and subscale.
    """
    accuracy = results.groupby(['subscale', 'level', 'thinking']).agg({
        'correct': ['mean', 'std', 'count']
    }).round(4)

    accuracy.columns = ['accuracy', 'std', 'n']

    return accuracy


def compute_moralchoice_consistency(results):
    """
    Compute consistency metrics for MoralChoice.

    Consistency = proportion of runs where model gives same answer.

    Returns:
        consistency_df: Per-item consistency scores
        summary: Aggregated by condition
    """
    def item_consistency(group):
        answers = group['extracted_answer'].dropna()
        if len(answers) < 2:
            return np.nan
        # Proportion of most common answer
        return answers.value_counts().iloc[0] / len(answers)

    consistency = results.groupby(['item_id', 'level', 'thinking']).apply(
        item_consistency
    ).reset_index(name='consistency')

    # Average by condition
    summary = consistency.groupby(['level', 'thinking'])['consistency'].agg(
        ['mean', 'std', 'count']
    ).round(4)
    summary.columns = ['consistency', 'std', 'n']

    return consistency, summary


def compute_moralchoice_consistency_by_ambiguity(results):
    """
    Compute consistency metrics by ambiguity level.
    """
    def item_consistency(group):
        answers = group['extracted_answer'].dropna()
        if len(answers) < 2:
            return np.nan
        return answers.value_counts().iloc[0] / len(answers)

    consistency = results.groupby(['item_id', 'level', 'thinking', 'ambiguity']).apply(
        item_consistency
    ).reset_index(name='consistency')

    summary = consistency.groupby(['ambiguity', 'level', 'thinking'])['consistency'].agg(
        ['mean', 'std']
    ).round(4)
    summary.columns = ['consistency', 'std']

    return consistency, summary


def two_way_anova(results, dv='correct'):
    """
    Run two-way ANOVA: prompt_level × thinking.

    Args:
        results: DataFrame with 'level', 'thinking', and dv columns
        dv: Dependent variable column name

    Returns:
        anova_table: ANOVA results
        model: Fitted OLS model
    """
    # Filter to valid responses
    valid = results[results[dv].notna()].copy()
    valid['level'] = valid['level'].astype(str)
    valid['thinking'] = valid['thinking'].astype(str)

    # Fit model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ols(f'{dv} ~ C(level) * C(thinking)', data=valid).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table, model


def trend_analysis(results, dv='correct'):
    """
    Test for monotonic trend across prompt levels.

    Uses Spearman correlation between level and mean accuracy.

    Returns:
        Dictionary with trend results for each thinking condition
    """
    trends = {}

    for thinking in [False, True]:
        subset = results[(results['thinking'] == thinking) & (results[dv].notna())]

        # Aggregate by level
        by_level = subset.groupby('level')[dv].mean()

        if len(by_level) < 3:
            trends[thinking] = {'spearman_r': np.nan, 'p_value': np.nan, 'values': {}}
            continue

        # Spearman correlation
        corr, p = spearmanr(by_level.index, by_level.values)

        trends[thinking] = {
            'spearman_r': round(corr, 4),
            'p_value': round(p, 4),
            'values': by_level.to_dict()
        }

    return trends


def pairwise_comparisons(results, dv='correct', alpha=0.05):
    """
    Run pairwise comparisons with multiple comparison correction.

    Compares:
    1. Adjacent levels within each thinking condition
    2. Thinking ON vs OFF within each level

    Returns DataFrame with chi-square test results and adjusted p-values.
    """
    comparisons = []
    valid = results[results[dv].notna()].copy()

    # Compare adjacent levels within each thinking condition
    for thinking in [False, True]:
        subset = valid[valid['thinking'] == thinking]
        thinking_label = "ON" if thinking else "OFF"

        for level in range(5):
            level_a_data = subset[subset['level'] == level]
            level_b_data = subset[subset['level'] == level + 1]

            if len(level_a_data) == 0 or len(level_b_data) == 0:
                continue

            # Create contingency table
            try:
                contingency = pd.crosstab(
                    pd.concat([level_a_data, level_b_data])['level'].astype(str),
                    pd.concat([level_a_data, level_b_data])[dv]
                )

                if contingency.shape == (2, 2):
                    chi2, p, dof, expected = chi2_contingency(contingency)
                else:
                    chi2, p = np.nan, np.nan
            except:
                chi2, p = np.nan, np.nan

            comparisons.append({
                'comparison': f'L{level} vs L{level+1}',
                'thinking': thinking_label,
                'chi2': round(chi2, 4) if not np.isnan(chi2) else np.nan,
                'p_value': round(p, 4) if not np.isnan(p) else np.nan,
                'mean_a': round(level_a_data[dv].mean(), 4),
                'mean_b': round(level_b_data[dv].mean(), 4),
                'diff': round(level_b_data[dv].mean() - level_a_data[dv].mean(), 4)
            })

    # Compare thinking ON vs OFF within each level
    for level in range(6):
        subset = valid[valid['level'] == level]

        off_data = subset[subset['thinking'] == False]
        on_data = subset[subset['thinking'] == True]

        if len(off_data) == 0 or len(on_data) == 0:
            continue

        try:
            contingency = pd.crosstab(
                subset['thinking'].astype(str),
                subset[dv]
            )

            if contingency.shape == (2, 2):
                chi2, p, dof, expected = chi2_contingency(contingency)
            else:
                chi2, p = np.nan, np.nan
        except:
            chi2, p = np.nan, np.nan

        comparisons.append({
            'comparison': f'L{level}: OFF vs ON',
            'thinking': 'comparison',
            'chi2': round(chi2, 4) if not np.isnan(chi2) else np.nan,
            'p_value': round(p, 4) if not np.isnan(p) else np.nan,
            'mean_a': round(off_data[dv].mean(), 4),
            'mean_b': round(on_data[dv].mean(), 4),
            'diff': round(on_data[dv].mean() - off_data[dv].mean(), 4)
        })

    comp_df = pd.DataFrame(comparisons)

    # Multiple comparison correction (FDR)
    if len(comp_df) > 0:
        valid_p = comp_df['p_value'].dropna()
        if len(valid_p) > 0:
            _, p_adj, _, _ = multipletests(valid_p, method='fdr_bh')
            comp_df.loc[comp_df['p_value'].notna(), 'p_adjusted'] = [round(p, 4) for p in p_adj]

    return comp_df


def preference_analysis(mc_results):
    """
    Analyze preference patterns in MoralChoice.

    Returns:
        preference: Preference rate (% choosing A) by condition
        kw_results: Kruskal-Wallis test results
    """
    valid = mc_results[mc_results['extracted_answer'].notna()].copy()

    # Compute preference rate (% choosing A) by condition
    preference = valid.groupby(['level', 'thinking']).apply(
        lambda x: (x['extracted_answer'] == 'A').mean()
    ).reset_index(name='pref_A')
    preference['pref_A'] = preference['pref_A'].round(4)

    # Kruskal-Wallis test across levels
    kw_results = {}
    for thinking in [False, True]:
        subset = valid[valid['thinking'] == thinking]
        groups = [
            subset[subset['level'] == level]['extracted_answer'].map({'A': 1, 'B': 0}).dropna()
            for level in range(6)
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                kw_results[thinking] = {'H': round(stat, 4), 'p': round(p, 4)}
            except:
                kw_results[thinking] = {'H': np.nan, 'p': np.nan}
        else:
            kw_results[thinking] = {'H': np.nan, 'p': np.nan}

    return preference, kw_results


def preference_analysis_by_ambiguity(mc_results):
    """
    Analyze preference patterns by ambiguity level.
    """
    valid = mc_results[mc_results['extracted_answer'].notna()].copy()

    preference = valid.groupby(['ambiguity', 'level', 'thinking']).apply(
        lambda x: (x['extracted_answer'] == 'A').mean()
    ).reset_index(name='pref_A')
    preference['pref_A'] = preference['pref_A'].round(4)

    return preference


def compute_response_metrics(results):
    """
    Compute response length and marker metrics by condition.
    """
    metrics = results.groupby(['level', 'thinking']).agg({
        'response_length': ['mean', 'std'],
        'reasoning_markers': ['mean', 'std'],
        'uncertainty_markers': ['mean', 'std'],
        'input_tokens': 'mean',
        'output_tokens': 'mean'
    }).round(2)

    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]

    return metrics


# =============================================================================
# MORABLES ANALYSIS FUNCTIONS
# =============================================================================

def compute_morables_accuracy(results):
    """
    Compute accuracy metrics for MORABLES.

    Returns DataFrame with accuracy, std, n, and extraction failure rate
    by level and thinking condition.
    """
    # Filter out error rows
    valid = results[results['extracted_answer'].notna() | results['correct'].notna()].copy()

    # Overall accuracy by condition
    accuracy = results.groupby(['level', 'thinking']).agg({
        'correct': ['mean', 'std', 'count'],
        'extracted_answer': lambda x: x.isna().mean()
    }).round(4)

    accuracy.columns = ['accuracy', 'std', 'n', 'extraction_failure']

    return accuracy


def compute_morables_consistency(results):
    """
    Compute consistency metrics for MORABLES (answer agreement across runs).

    Consistency = proportion of runs where model gives same answer.

    Returns:
        consistency_df: Per-item consistency scores
        summary: Aggregated by condition
    """
    def item_consistency(group):
        answers = group['extracted_answer'].dropna()
        if len(answers) < 2:
            return np.nan
        # Proportion of most common answer
        return answers.value_counts().iloc[0] / len(answers)

    consistency = results.groupby(['item_id', 'level', 'thinking']).apply(
        item_consistency
    ).reset_index(name='consistency')

    # Average by condition
    summary = consistency.groupby(['level', 'thinking'])['consistency'].agg(
        ['mean', 'std', 'count']
    ).round(4)
    summary.columns = ['consistency', 'std', 'n']

    return consistency, summary


def compute_morables_answer_distribution(results):
    """
    Compute distribution of answers (A-E) by condition.

    Useful for detecting if model has preference for certain options.
    """
    valid = results[results['extracted_answer'].notna()].copy()

    # Count by answer and condition
    distribution = valid.groupby(['level', 'thinking', 'extracted_answer']).size().unstack(fill_value=0)

    # Normalize to percentages
    distribution_pct = distribution.div(distribution.sum(axis=1), axis=0).round(4)

    return distribution, distribution_pct


def compute_morables_self_contradiction(results):
    """
    Compute self-contradiction rate for MORABLES.

    Self-contradiction = proportion of items where model gives different answers
    across identical prompts (multiple runs).

    The MORABLES paper reports ~20% self-contradiction rate.
    """
    def item_contradiction(group):
        answers = group['extracted_answer'].dropna()
        if len(answers) < 2:
            return np.nan
        # If all answers are the same, no contradiction
        unique_answers = answers.nunique()
        return 1 if unique_answers > 1 else 0

    contradictions = results.groupby(['item_id', 'level', 'thinking']).apply(
        item_contradiction
    ).reset_index(name='self_contradiction')

    # Rate by condition
    summary = contradictions.groupby(['level', 'thinking'])['self_contradiction'].agg(
        ['mean', 'std', 'count']
    ).round(4)
    summary.columns = ['contradiction_rate', 'std', 'n']

    return contradictions, summary


def morables_distractor_analysis(results, correct_idx_col='correct_idx'):
    """
    Analyze which answer positions models choose when incorrect.

    Useful for understanding if certain distractor positions are more common errors.
    """
    valid = results[results['extracted_answer'].notna()].copy()

    # Map letters to positions
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    valid['answer_idx'] = valid['extracted_answer'].map(letter_to_idx)

    # Filter to incorrect answers
    incorrect = valid[valid['correct'] == False].copy()

    if len(incorrect) == 0:
        return None

    # Count by answer position
    error_distribution = incorrect.groupby(['level', 'thinking', 'extracted_answer']).size().unstack(fill_value=0)

    return error_distribution


# =============================================================================
# CONFIDENCE ANALYSIS FUNCTIONS
# =============================================================================

def compute_confidence_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean confidence by level and thinking condition.

    Args:
        df: Results DataFrame with 'confidence' column

    Returns:
        DataFrame with confidence statistics by condition
    """
    if 'confidence' not in df.columns:
        return None

    valid = df[df['confidence'].notna()].copy()

    if len(valid) == 0:
        return None

    confidence_stats = valid.groupby(['level', 'thinking']).agg(
        mean_confidence=('confidence', 'mean'),
        std_confidence=('confidence', 'std'),
        median_confidence=('confidence', 'median'),
        n=('confidence', 'count')
    ).round(2)

    return confidence_stats


def compute_confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibration: does high confidence predict correct answers?

    Returns DataFrame with confidence bins and accuracy within each bin.
    """
    if 'confidence' not in df.columns or 'correct' not in df.columns:
        return None

    # Filter to items with both confidence and correctness
    valid = df[df['confidence'].notna() & df['correct'].notna()].copy()

    if len(valid) == 0:
        return None

    # Bin confidence scores
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    valid['confidence_bin'] = pd.cut(valid['confidence'], bins=bins, labels=labels)

    calibration = valid.groupby(['level', 'thinking', 'confidence_bin']).agg(
        accuracy=('correct', 'mean'),
        n=('correct', 'count'),
        mean_confidence=('confidence', 'mean')
    ).round(4)

    return calibration


def compute_overconfidence_rate(df: pd.DataFrame, threshold: int = 70) -> pd.DataFrame:
    """
    Compute overconfidence: high confidence (>threshold) but incorrect.

    Args:
        df: Results DataFrame
        threshold: Confidence threshold for "high confidence" (default 70)

    Returns:
        DataFrame with overconfidence rates by condition
    """
    if 'confidence' not in df.columns or 'correct' not in df.columns:
        return None

    valid = df[df['confidence'].notna() & df['correct'].notna()].copy()

    if len(valid) == 0:
        return None

    valid['overconfident_error'] = (valid['confidence'] > threshold) & (valid['correct'] == False)
    valid['high_confidence'] = valid['confidence'] > threshold

    rates = valid.groupby(['level', 'thinking']).agg(
        overconfident_error_rate=('overconfident_error', 'mean'),
        high_confidence_rate=('high_confidence', 'mean'),
        n=('correct', 'count')
    ).round(4)

    return rates


def confidence_accuracy_correlation(df: pd.DataFrame) -> dict:
    """
    Compute correlation between confidence and accuracy.

    Returns Spearman correlation and p-value.
    """
    if 'confidence' not in df.columns or 'correct' not in df.columns:
        return {'correlation': None, 'p_value': None, 'n': 0}

    valid = df[df['confidence'].notna() & df['correct'].notna()].copy()

    if len(valid) < 10:
        return {'correlation': None, 'p_value': None, 'n': len(valid)}

    corr, p_value = spearmanr(valid['confidence'], valid['correct'].astype(int))

    return {
        'correlation': round(corr, 4),
        'p_value': round(p_value, 4),
        'n': len(valid)
    }


def compare_confidence_correct_vs_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare confidence distributions for correct vs incorrect answers.

    Tests: Are models more confident when correct?
    """
    from scipy.stats import mannwhitneyu

    if 'confidence' not in df.columns or 'correct' not in df.columns:
        return None

    valid = df[df['confidence'].notna() & df['correct'].notna()].copy()

    results = []
    for (level, thinking), group in valid.groupby(['level', 'thinking']):
        correct = group[group['correct'] == True]['confidence']
        incorrect = group[group['correct'] == False]['confidence']

        if len(correct) >= 5 and len(incorrect) >= 5:
            stat, p_value = mannwhitneyu(correct, incorrect, alternative='greater')
        else:
            stat, p_value = None, None

        results.append({
            'level': level,
            'thinking': thinking,
            'mean_conf_correct': correct.mean() if len(correct) > 0 else None,
            'mean_conf_incorrect': incorrect.mean() if len(incorrect) > 0 else None,
            'diff': (correct.mean() - incorrect.mean()) if len(correct) > 0 and len(incorrect) > 0 else None,
            'mann_whitney_stat': stat,
            'p_value': p_value,
            'n_correct': len(correct),
            'n_incorrect': len(incorrect)
        })

    return pd.DataFrame(results)


def run_confidence_analyses(df: pd.DataFrame, benchmark_name: str = "benchmark") -> dict:
    """
    Run all confidence-related analyses for a dataset.

    Args:
        df: Results DataFrame with 'confidence' and optionally 'correct' columns
        benchmark_name: Name for labeling output

    Returns:
        Dictionary with all confidence analysis results
    """
    results = {}

    print(f"\n{'=' * 60}")
    print(f"CONFIDENCE ANALYSIS - {benchmark_name.upper()}")
    print("=" * 60)

    # 1. Confidence by condition
    print(f"\n1. {benchmark_name.upper()} CONFIDENCE BY CONDITION")
    print("-" * 40)
    conf_by_cond = compute_confidence_by_condition(df)
    if conf_by_cond is not None:
        print(conf_by_cond)
        results['confidence_by_condition'] = conf_by_cond
    else:
        print("No confidence data available")

    # 2. Calibration (only for benchmarks with ground truth)
    if 'correct' in df.columns:
        print(f"\n2. {benchmark_name.upper()} CALIBRATION")
        print("-" * 40)
        calibration = compute_confidence_calibration(df)
        if calibration is not None:
            # Show summary
            print("Accuracy by confidence bin (averaged across conditions):")
            summary = df[df['confidence'].notna() & df['correct'].notna()].copy()
            bins = [0, 20, 40, 60, 80, 100]
            labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
            summary['conf_bin'] = pd.cut(summary['confidence'], bins=bins, labels=labels)
            cal_summary = summary.groupby('conf_bin')['correct'].agg(['mean', 'count'])
            print(cal_summary.round(3))
            results['calibration'] = calibration
        else:
            print("Insufficient data for calibration analysis")

        # 3. Overconfidence
        print(f"\n3. {benchmark_name.upper()} OVERCONFIDENCE RATE")
        print("-" * 40)
        overconf = compute_overconfidence_rate(df)
        if overconf is not None:
            print(overconf)
            results['overconfidence'] = overconf
        else:
            print("Insufficient data for overconfidence analysis")

        # 4. Confidence-accuracy correlation
        print(f"\n4. {benchmark_name.upper()} CONFIDENCE-ACCURACY CORRELATION")
        print("-" * 40)
        corr_result = confidence_accuracy_correlation(df)
        print(f"Spearman r = {corr_result['correlation']}, p = {corr_result['p_value']}, n = {corr_result['n']}")
        results['correlation'] = corr_result

        # 5. Confidence by correctness
        print(f"\n5. {benchmark_name.upper()} CONFIDENCE BY CORRECTNESS")
        print("-" * 40)
        conf_by_correct = compare_confidence_correct_vs_incorrect(df)
        if conf_by_correct is not None:
            print(conf_by_correct[['level', 'thinking', 'mean_conf_correct', 'mean_conf_incorrect', 'diff', 'p_value']].round(3))
            results['confidence_by_correctness'] = conf_by_correct
        else:
            print("Insufficient data for analysis")

    return results


def run_all_analyses(ethics_results, mc_results, morables_results=None):
    """
    Run complete analysis suite.

    Returns dictionary with all analysis results.
    """
    print("=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    results = {}

    # 1. ETHICS Accuracy
    print("\n1. ETHICS ACCURACY BY CONDITION")
    print("-" * 40)
    accuracy = compute_ethics_accuracy(ethics_results)
    print(accuracy)
    results['ethics_accuracy'] = accuracy

    # 2. ETHICS Accuracy by Subscale
    print("\n2. ETHICS ACCURACY BY SUBSCALE")
    print("-" * 40)
    subscale_acc = compute_ethics_accuracy_by_subscale(ethics_results)
    print(subscale_acc)
    results['ethics_accuracy_subscale'] = subscale_acc

    # 3. Two-way ANOVA
    print("\n3. TWO-WAY ANOVA (level × thinking)")
    print("-" * 40)
    try:
        anova_table, model = two_way_anova(ethics_results)
        print(anova_table)
        results['anova'] = anova_table
    except Exception as e:
        print(f"ANOVA failed: {e}")
        results['anova'] = None

    # 4. Trend analysis
    print("\n4. TREND ANALYSIS")
    print("-" * 40)
    trends = trend_analysis(ethics_results)
    for thinking, result in trends.items():
        label = "ON" if thinking else "OFF"
        print(f"Thinking {label}: r={result['spearman_r']}, p={result['p_value']}")
    results['trends'] = trends

    # 5. Pairwise comparisons
    print("\n5. PAIRWISE COMPARISONS")
    print("-" * 40)
    comparisons = pairwise_comparisons(ethics_results)
    if 'p_adjusted' in comparisons.columns:
        sig_comparisons = comparisons[comparisons['p_adjusted'] < 0.05]
        print(f"Significant comparisons (p_adj < 0.05): {len(sig_comparisons)}")
        if len(sig_comparisons) > 0:
            print(sig_comparisons[['comparison', 'thinking', 'diff', 'p_adjusted']])
    results['comparisons'] = comparisons

    # 6. MoralChoice consistency
    print("\n6. MORALCHOICE CONSISTENCY")
    print("-" * 40)
    consistency_items, consistency_summary = compute_moralchoice_consistency(mc_results)
    print(consistency_summary)
    results['mc_consistency'] = consistency_summary
    results['mc_consistency_items'] = consistency_items

    # 7. MoralChoice consistency by ambiguity
    print("\n7. MORALCHOICE CONSISTENCY BY AMBIGUITY")
    print("-" * 40)
    _, consistency_ambiguity = compute_moralchoice_consistency_by_ambiguity(mc_results)
    print(consistency_ambiguity)
    results['mc_consistency_ambiguity'] = consistency_ambiguity

    # 8. Preference analysis
    print("\n8. MORALCHOICE PREFERENCE ANALYSIS")
    print("-" * 40)
    preference, kw_results = preference_analysis(mc_results)
    print("Preference rate (% choosing A):")
    print(preference.pivot(index='level', columns='thinking', values='pref_A').round(3))
    print("\nKruskal-Wallis tests:")
    for thinking, result in kw_results.items():
        label = "ON" if thinking else "OFF"
        print(f"  Thinking {label}: H={result['H']}, p={result['p']}")
    results['mc_preference'] = preference
    results['mc_kruskal'] = kw_results

    # 9. Response metrics
    print("\n9. RESPONSE METRICS")
    print("-" * 40)
    ethics_metrics = compute_response_metrics(ethics_results)
    print("ETHICS response metrics:")
    print(ethics_metrics[['response_length_mean', 'reasoning_markers_mean', 'uncertainty_markers_mean']])
    results['ethics_metrics'] = ethics_metrics

    # 10-13. MORABLES analyses (if provided)
    if morables_results is not None:
        print("\n" + "=" * 60)
        print("MORABLES ANALYSIS")
        print("=" * 60)

        # 10. MORABLES Accuracy
        print("\n10. MORABLES ACCURACY BY CONDITION")
        print("-" * 40)
        morables_accuracy = compute_morables_accuracy(morables_results)
        print(morables_accuracy)
        results['morables_accuracy'] = morables_accuracy

        # 11. MORABLES Consistency
        print("\n11. MORABLES CONSISTENCY")
        print("-" * 40)
        morables_consistency_items, morables_consistency_summary = compute_morables_consistency(morables_results)
        print(morables_consistency_summary)
        results['morables_consistency'] = morables_consistency_summary
        results['morables_consistency_items'] = morables_consistency_items

        # 12. MORABLES Self-contradiction
        print("\n12. MORABLES SELF-CONTRADICTION RATE")
        print("-" * 40)
        _, contradiction_summary = compute_morables_self_contradiction(morables_results)
        print(contradiction_summary)
        results['morables_contradiction'] = contradiction_summary

        # 13. MORABLES Answer Distribution
        print("\n13. MORABLES ANSWER DISTRIBUTION")
        print("-" * 40)
        _, dist_pct = compute_morables_answer_distribution(morables_results)
        print("Answer distribution (%):")
        print(dist_pct.round(3))
        results['morables_answer_dist'] = dist_pct

        # 14. MORABLES Response metrics
        print("\n14. MORABLES RESPONSE METRICS")
        print("-" * 40)
        morables_metrics = compute_response_metrics(morables_results)
        print(morables_metrics[['response_length_mean', 'reasoning_markers_mean', 'uncertainty_markers_mean']])
        results['morables_metrics'] = morables_metrics

    return results
