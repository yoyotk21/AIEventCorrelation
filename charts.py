import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.8,
})

# Chart 1: Feature Weights
features = [
    "Temporal overlap",
    "Resolution proximity",
    "Granger causality",
    "Price correlation",
    "Volume spike",
    "Same category",
    "Tag Jaccard",
]
weights = [0.8139, 0.1561, 0.0300, 0.0000, 0.0000, 0.0000, 0.0000]

# Sort by weight descending
order = np.argsort(weights)[::-1]
features = [features[i] for i in order]
weights  = [weights[i]  for i in order]

LIGHT_PURPLE = "#C8C5EE"
DARK_PURPLE  = "#7F77DD"
colors = [DARK_PURPLE if w > 0.05 else LIGHT_PURPLE for w in weights]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.grid(axis="x")
bars = ax.barh(features[::-1], weights[::-1], color=colors[::-1], height=0.55)

# Value labels on bars
for bar, w in zip(bars, weights[::-1]):
    ax.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.4f}",
        va="center", ha="left", fontsize=11,
        color="black" if w > 0.05 else "#888"
    )

ax.set_xlim(0, 1.05)
ax.set_xlabel("Learned weight", fontsize=12)
ax.set_title("Optimized feature weights (hill climbing)", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
plt.savefig("chart_weights.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_weights.png")

# ── Chart 2: MSE Comparison ───────────────────────────────────────────────────
LIGHT_PURPLE = "#7E77BB"
DARK_PURPLE = "#3C3489"

models  = ["Random\nweights", "Equal\nweights", "Price\nonly", "Linear\nregression", "Optimized\n(ours)"]
mse     = [0.6117, 0.5525, 0.8109, 0.1977, 0.2006]
colors2 = [LIGHT_PURPLE, LIGHT_PURPLE, LIGHT_PURPLE, LIGHT_PURPLE, DARK_PURPLE]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars2 = ax.bar(models, mse, color=colors2, width=0.5)

# Value labels on top of bars
for bar, m in zip(bars2, mse):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{m:.3f}",
        ha="center", va="bottom", fontsize=11,
        color="#3C3489" if bar.get_facecolor() == (0.498, 0.467, 0.867, 1.0) else "#888"
    )

ax.set_ylim(0, 1.0)
ax.set_ylabel("MSE (lower is better)", fontsize=12)
ax.set_title("MSE comparison across baselines", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y")
ax.grid(axis="x", visible=False)

# Annotation arrow pointing to our model
ax.annotate(
    "Our model",
    xy=(4, 0.2006),
    xytext=(3.2, 0.55),
    arrowprops=dict(arrowstyle="->", color="#3C3489", lw=1.5),
    fontsize=11, color="#3C3489", fontweight="bold"
)

plt.tight_layout()
plt.savefig("chart_mse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_mse.png")