import pandas as pd
import numpy as np
from scipy.stats import t, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "itunes"

RESULTS_PATH = f"results/{dataset}"

# Load your results
my_results = pd.read_csv(f"{RESULTS_PATH}/results_summary.csv")

print(my_results)

# Data from the paper
paper_results = {
    "deezer": {
        20: {
            (0.2, 0.3): 50.2,
            (0.2, 0.4): 53.1,
            (0.3, 0.3): 48.5,
            (0.3, 0.4): 53.3,
            (0.4, 0.3): 53.2,
            (0.4, 0.4): 56.4,
            (0.5, 0.3): 52.5,
            (0.5, 0.4): 56.3,
        },
        50: {
            (0.2, 0.3): 48.9,
            (0.2, 0.4): 49.2,
            (0.3, 0.3): 52.1,
            (0.3, 0.4): 50.9,
            (0.4, 0.3): 51.5,
            (0.4, 0.4): 52.6,
            (0.5, 0.3): 56.3,
            (0.5, 0.4): 60.6,
        },
        100: {
            (0.2, 0.3): 51.4,
            (0.2, 0.4): 50.8,
            (0.3, 0.3): 51.5,
            (0.3, 0.4): 55.3,
            (0.4, 0.3): 52.2,
            (0.4, 0.4): 48.1,
            (0.5, 0.3): 50.8,
            (0.5, 0.4): 54.9,
        },
        200: {
            (0.2, 0.3): 48.4,
            (0.2, 0.4): 50.6,
            (0.3, 0.3): 49.8,
            (0.3, 0.4): 51.6,
            (0.4, 0.3): 50.0,
            (0.4, 0.4): 49.0,
            (0.5, 0.3): 55.4,
            (0.5, 0.4): 53.3,
        },
    },
    "itunes": {
        20: {
            (0.2, 0.3): 49.3,
            (0.2, 0.4): 47.2,
            (0.3, 0.3): 50.3,
            (0.3, 0.4): 52.5,
            (0.4, 0.3): 52.8,
            (0.4, 0.4): 52.4,
            (0.5, 0.3): 50.6,
            (0.5, 0.4): 50.5,
        },
        50: {
            (0.2, 0.3): 43.3,
            (0.2, 0.4): 49.5,
            (0.3, 0.3): 52.5,
            (0.3, 0.4): 49.5,
            (0.4, 0.3): 50.1,
            (0.4, 0.4): 51.9,
            (0.5, 0.3): 46.5,
            (0.5, 0.4): 52.0,
        },
        100: {
            (0.2, 0.3): 49.5,
            (0.2, 0.4): 50.7,
            (0.3, 0.3): 49.0,
            (0.3, 0.4): 49.2,
            (0.4, 0.3): 50.6,
            (0.4, 0.4): 49.9,
            (0.5, 0.3): 46.7,
            (0.5, 0.4): 48.7,
        },
        200: {
            (0.2, 0.3): 47.0,
            (0.2, 0.4): 51.3,
            (0.3, 0.3): 48.2,
            (0.3, 0.4): 49.8,
            (0.4, 0.3): 51.1,
            (0.4, 0.4): 47.4,
            (0.5, 0.3): 49.0,
            (0.5, 0.4): 46.1,
        },
    },
}


# Function to calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    ci = t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return mean, ci


# Initialize results storage
comparison_results = []

# Group your results by K, alpha_word, and alpha_ent
grouped = my_results.groupby(["K", "alpha_word", "alpha_ent"])["evaluation_result"]

for (k, alpha_word, alpha_ent), group in grouped:
    # Mean and confidence intervals for your results
    mean, ci = confidence_interval(group)
    variance = np.var(group)

    print(group, variance)

    # Get the corresponding value from the paper
    paper_result = paper_results[dataset][k].get((alpha_word, alpha_ent))

    if paper_result is not None:
        # Perform t-test to check if your results differ from the paper's
        t_stat, p_value = ttest_1samp(group, popmean=paper_result)

        # Store results
        comparison_results.append(
            {
                "K": k,
                "alpha_word": alpha_word,
                "alpha_ent": alpha_ent,
                "mean": mean,
                "ci": ci,
                "variance": variance,
                "paper_result": paper_result,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

# Convert to DataFrame for easy analysis
comparison_df = pd.DataFrame(comparison_results)

# Save comparison results to a CSV file
comparison_df.to_csv(f"{RESULTS_PATH}/comparison_results.csv", index=False)

# Print results
print(comparison_df)

# Bar Plot: Mean vs Paper Result
hues = ["alpha_word", "alpha_ent"]
y_values = ["mean", "paper_result"]

for hue in hues:
    for y_var in y_values:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=comparison_df, x="K", y=y_var, hue=hue, palette="viridis")
        plt.title(f"{y_var} by K and {hue}")
        plt.ylabel(y_var)
        plt.xlabel("K")
        plt.legend(title=hue)
        plt.savefig(f"{RESULTS_PATH}/{y_var}_K_{hue}.png")


# Heatmap: Mean-Paper Differences
comparison_df["mean_paper_diff"] = comparison_df["mean"] - comparison_df["paper_result"]
print(comparison_df["mean_paper_diff"])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# for each K value, plot a heatmap

for i, k in enumerate(comparison_df["K"].unique()):
    heatmap_data = comparison_df[comparison_df["K"] == k].pivot(
        index="alpha_word", columns="alpha_ent", values="mean_paper_diff"
    )

    subplot = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        center=0,
        ax=axes[i // 2, i % 2],
        annot_kws={"fontsize": 15},
    )
    subplot.collections[0].colorbar.ax.tick_params(labelsize=15)
    # subplot.collections[0].colorbar.ax.set_ylabel("Mean Difference", fontsize=15)
    subplot.set_title(f"Mean Difference for K={k}", fontsize=20)
    subplot.set_xlabel("alpha_ent", fontsize=15)
    subplot.set_ylabel("alpha_word", fontsize=15)
    subplot.set_xticklabels(subplot.get_xticklabels(), fontsize=15)
    subplot.set_yticklabels(subplot.get_yticklabels(), fontsize=15)


plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/mean_paper_diff_heatmap.png")


# Heatmap: P-Values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# for each K value, plot a heatmap
for i, k in enumerate(comparison_df["K"].unique()):
    heatmap_data = comparison_df[comparison_df["K"] == k].pivot(
        index="alpha_word", columns="alpha_ent", values="p_value"
    )

    subplot = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        # cbar_kws={"label": "p-value"},
        annot_kws={"fontsize": 15},
        ax=axes[i // 2, i % 2],
    )
    subplot.collections[0].colorbar.ax.tick_params(labelsize=15)
    # subplot.collections[0].colorbar.ax.set_ylabel("p-value", fontsize=15)
    subplot.set_xticklabels(subplot.get_xticklabels(), fontsize=15)
    subplot.set_yticklabels(subplot.get_yticklabels(), fontsize=15)
    subplot.set_title(f"P-Values for K={k}", fontsize=20)
    subplot.set_xlabel("alpha_ent", fontsize=15)
    subplot.set_ylabel("alpha_word", fontsize=15)

plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/p_value_heatmap.png")
