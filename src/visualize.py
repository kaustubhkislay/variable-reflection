"""Visualization for reflection study."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'OFF': '#1f77b4', 'ON': '#ff7f0e'}
LEVEL_LABELS = ['0\n(Direct)', '1\n(Minimal)', '2\n(CoT)',
                '3\n(Structured)', '4\n(Adversarial)', '5\n(Two-Pass)']


def plot_accuracy_by_condition(results, save_path=None):
    """Plot accuracy by prompt level and thinking condition."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = results[results['thinking'] == thinking]
        means = subset.groupby('level')['correct'].mean()
        sems = subset.groupby('level')['correct'].sem()

        ax.errorbar(
            means.index, means.values,
            yerr=1.96 * sems.values,
            label=label, color=color,
            marker='o', markersize=8, linewidth=2, capsize=5
        )

    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('ETHICS Accuracy by Reflection Condition', fontsize=14)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_accuracy_heatmap(results, save_path=None):
    """Heatmap of accuracy by level and thinking."""

    pivot = results.groupby(['level', 'thinking'])['correct'].mean().reset_index()
    pivot['thinking'] = pivot['thinking'].map({False: 'OFF', True: 'ON'})
    heatmap_data = pivot.pivot(index='level', columns='thinking', values='correct')

    fig, ax = plt.subplots(figsize=(6, 8))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Accuracy'}
    )

    ax.set_xlabel('Extended Thinking', fontsize=12)
    ax.set_ylabel('Prompt Level', fontsize=12)
    ax.set_title('ETHICS Accuracy Heatmap', fontsize=14)
    ax.set_yticklabels(['0 (Direct)', '1 (Minimal)', '2 (CoT)',
                        '3 (Structured)', '4 (Adversarial)', '5 (Two-Pass)'],
                       rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_subscale_accuracy(results, save_path=None):
    """Plot accuracy by subscale and condition."""

    subscales = results['subscale'].unique()
    n_subscales = len(subscales)

    fig, axes = plt.subplots(1, n_subscales, figsize=(5*n_subscales, 5), sharey=True)

    if n_subscales == 1:
        axes = [axes]

    for ax, subscale in zip(axes, subscales):
        subset = results[results['subscale'] == subscale]

        for thinking in [False, True]:
            label = 'ON' if thinking else 'OFF'
            color = COLORS['ON'] if thinking else COLORS['OFF']

            sub_subset = subset[subset['thinking'] == thinking]
            means = sub_subset.groupby('level')['correct'].mean()
            sems = sub_subset.groupby('level')['correct'].sem()

            ax.errorbar(
                means.index, means.values,
                yerr=1.96 * sems.values,
                label=f'Thinking {label}', color=color,
                marker='o', linewidth=2, capsize=3
            )

        ax.set_title(subscale.capitalize(), fontsize=12)
        ax.set_xlabel('Prompt Level')
        ax.set_xticks(range(6))
        ax.set_ylim(0.3, 1.0)

        if ax == axes[0]:
            ax.set_ylabel('Accuracy')
            ax.legend(loc='lower right')

    plt.suptitle('ETHICS Accuracy by Subscale', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_response_length(results, save_path=None):
    """Plot response length by condition."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = results[results['thinking'] == thinking]
        means = subset.groupby('level')['response_length'].mean()
        sems = subset.groupby('level')['response_length'].sem()

        ax.errorbar(
            means.index, means.values,
            yerr=1.96 * sems.values,
            label=label, color=color,
            marker='s', linewidth=2, capsize=5
        )

    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Response Length (words)', fontsize=12)
    ax.set_title('Response Length by Condition', fontsize=14)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_reasoning_markers(results, save_path=None):
    """Plot reasoning and uncertainty markers by condition."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reasoning markers
    ax = axes[0]
    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = results[results['thinking'] == thinking]
        means = subset.groupby('level')['reasoning_markers'].mean()

        ax.plot(means.index, means.values, label=label, color=color,
                marker='o', linewidth=2, markersize=8)

    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Reasoning Markers (count)', fontsize=12)
    ax.set_title('Reasoning Markers by Condition', fontsize=12)
    ax.set_xticks(range(6))
    ax.legend()

    # Uncertainty markers
    ax = axes[1]
    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = results[results['thinking'] == thinking]
        means = subset.groupby('level')['uncertainty_markers'].mean()

        ax.plot(means.index, means.values, label=label, color=color,
                marker='o', linewidth=2, markersize=8)

    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Uncertainty Markers (count)', fontsize=12)
    ax.set_title('Uncertainty Markers by Condition', fontsize=12)
    ax.set_xticks(range(6))
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_moralchoice_consistency(consistency_df, save_path=None):
    """Plot MoralChoice consistency by condition."""

    fig, ax = plt.subplots(figsize=(10, 6))

    summary = consistency_df.groupby(['level', 'thinking'])['consistency'].mean().reset_index()

    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = summary[summary['thinking'] == thinking]
        ax.plot(
            subset['level'], subset['consistency'],
            label=label, color=color,
            marker='o', linewidth=2, markersize=8
        )

    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Consistency (across runs)', fontsize=12)
    ax.set_title('MoralChoice Response Consistency', fontsize=14)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0.5, 1.05)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_moralchoice_consistency_by_ambiguity(mc_results, save_path=None):
    """Plot MoralChoice consistency by ambiguity level."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ambiguity in zip(axes, ['low', 'high']):
        subset = mc_results[mc_results['ambiguity'] == ambiguity]

        # Calculate consistency per item
        def item_consistency(group):
            answers = group['extracted_answer'].dropna()
            if len(answers) < 2:
                return np.nan
            return answers.value_counts().iloc[0] / len(answers)

        consistency = subset.groupby(['item_id', 'level', 'thinking']).apply(
            item_consistency
        ).reset_index(name='consistency')

        summary = consistency.groupby(['level', 'thinking'])['consistency'].mean().reset_index()

        for thinking in [False, True]:
            label = 'Thinking ON' if thinking else 'Thinking OFF'
            color = COLORS['ON'] if thinking else COLORS['OFF']

            sub = summary[summary['thinking'] == thinking]
            ax.plot(sub['level'], sub['consistency'], label=label, color=color,
                    marker='o', linewidth=2, markersize=8)

        ax.set_xlabel('Prompt Level', fontsize=12)
        ax.set_ylabel('Consistency', fontsize=12)
        ax.set_title(f'{ambiguity.capitalize()} Ambiguity Dilemmas', fontsize=12)
        ax.set_xticks(range(6))
        ax.set_ylim(0.5, 1.05)
        ax.legend()

    plt.suptitle('MoralChoice Consistency by Ambiguity', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_preference_shift(mc_results, save_path=None):
    """Plot preference rate across conditions."""

    fig, ax = plt.subplots(figsize=(10, 6))

    valid = mc_results[mc_results['extracted_answer'].notna()]

    for thinking in [False, True]:
        label = 'Thinking ON' if thinking else 'Thinking OFF'
        color = COLORS['ON'] if thinking else COLORS['OFF']

        subset = valid[valid['thinking'] == thinking]
        pref_rate = subset.groupby('level').apply(
            lambda x: (x['extracted_answer'] == 'A').mean()
        )

        ax.plot(
            pref_rate.index, pref_rate.values,
            label=label, color=color,
            marker='o', linewidth=2, markersize=8
        )

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='No preference')
    ax.set_xlabel('Prompt Level', fontsize=12)
    ax.set_ylabel('Preference Rate (% choosing A)', fontsize=12)
    ax.set_title('MoralChoice Preference by Condition', fontsize=14)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0.3, 0.7)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_preference_by_ambiguity(mc_results, save_path=None):
    """Plot preference rate by ambiguity level."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = mc_results[mc_results['extracted_answer'].notna()]

    for ax, ambiguity in zip(axes, ['low', 'high']):
        subset = valid[valid['ambiguity'] == ambiguity]

        for thinking in [False, True]:
            label = 'Thinking ON' if thinking else 'Thinking OFF'
            color = COLORS['ON'] if thinking else COLORS['OFF']

            sub = subset[subset['thinking'] == thinking]
            pref_rate = sub.groupby('level').apply(
                lambda x: (x['extracted_answer'] == 'A').mean()
            )

            ax.plot(pref_rate.index, pref_rate.values, label=label, color=color,
                    marker='o', linewidth=2, markersize=8)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prompt Level', fontsize=12)
        ax.set_ylabel('Preference Rate (% A)', fontsize=12)
        ax.set_title(f'{ambiguity.capitalize()} Ambiguity', fontsize=12)
        ax.set_xticks(range(6))
        ax.set_ylim(0.2, 0.8)
        ax.legend()

    plt.suptitle('MoralChoice Preference by Ambiguity', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_token_usage(ethics_results, mc_results, save_path=None):
    """Plot token usage by condition."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, title) in zip(axes, [(ethics_results, 'ETHICS'), (mc_results, 'MoralChoice')]):
        for thinking in [False, True]:
            label = 'Thinking ON' if thinking else 'Thinking OFF'
            color = COLORS['ON'] if thinking else COLORS['OFF']

            subset = results[results['thinking'] == thinking]
            means = subset.groupby('level')['output_tokens'].mean()

            ax.plot(means.index, means.values, label=label, color=color,
                    marker='o', linewidth=2, markersize=8)

        ax.set_xlabel('Prompt Level', fontsize=12)
        ax.set_ylabel('Output Tokens (mean)', fontsize=12)
        ax.set_title(f'{title} Token Usage', fontsize=12)
        ax.set_xticks(range(6))
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def create_all_figures(ethics_results, mc_results, consistency_df, output_dir='outputs/figures'):
    """Generate and save all figures."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    plot_accuracy_by_condition(ethics_results, f"{output_dir}/accuracy_by_condition.png")
    print("  - accuracy_by_condition.png")

    plot_accuracy_heatmap(ethics_results, f"{output_dir}/accuracy_heatmap.png")
    print("  - accuracy_heatmap.png")

    plot_subscale_accuracy(ethics_results, f"{output_dir}/subscale_accuracy.png")
    print("  - subscale_accuracy.png")

    plot_response_length(ethics_results, f"{output_dir}/response_length.png")
    print("  - response_length.png")

    plot_reasoning_markers(ethics_results, f"{output_dir}/reasoning_markers.png")
    print("  - reasoning_markers.png")

    plot_moralchoice_consistency(consistency_df, f"{output_dir}/moralchoice_consistency.png")
    print("  - moralchoice_consistency.png")

    plot_moralchoice_consistency_by_ambiguity(mc_results, f"{output_dir}/consistency_by_ambiguity.png")
    print("  - consistency_by_ambiguity.png")

    plot_preference_shift(mc_results, f"{output_dir}/preference_shift.png")
    print("  - preference_shift.png")

    plot_preference_by_ambiguity(mc_results, f"{output_dir}/preference_by_ambiguity.png")
    print("  - preference_by_ambiguity.png")

    plot_token_usage(ethics_results, mc_results, f"{output_dir}/token_usage.png")
    print("  - token_usage.png")

    print(f"\nAll figures saved to {output_dir}/")
