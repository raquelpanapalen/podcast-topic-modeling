import pandas as pd
import numpy as np
from scipy.stats import t, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt

# Load your results
my_results = pd.read_csv("results/results_summary.csv")

# Data from the paper
paper_results = {
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
    variance = np.var(group, ddof=1)

    # Get the corresponding value from the paper
    paper_result = paper_results.get(k, {}).get((alpha_word, alpha_ent))

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
comparison_df.to_csv("results/comparison_results.csv", index=False)

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
        plt.savefig(f"results/{y_var}_K_{hue}.png")


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
        heatmap_data, annot=True, cmap="coolwarm", center=0, ax=axes[i // 2, i % 2]
    )
    subplot.set_title(f"Mean-Paper Result Differences for K={k}")

plt.tight_layout()
plt.savefig("results/mean_paper_diff_heatmap.png")
