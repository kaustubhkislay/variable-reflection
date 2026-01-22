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


def run_all_analyses(ethics_results, mc_results):
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

    return results
