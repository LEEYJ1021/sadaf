import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"]
mpl.rcParams["axes.unicode_minus"] = False

K = {
    "navy":   "#1A3A5C",
    "teal":   "#2E7D6E",
    "grey":   "#8C8C8C",
    "lgrey":  "#D3D3D3",
    "border": "#1A1A1A",
    "text":   "#111111",
    "sub":    "#3A3A3A",
    "accent": "#C0392B",
}

seeds = [42, 1, 7, 123, 2024]

data = {
    "BayesianLSTM": {"vals": [1.3420, 1.3775, 1.3281, 1.3713, 1.3829], "mean": 1.3604, "sd": 0.0240, "rank1": 40.0},
    "GRU":          {"vals": [1.3984, 1.4784, 1.4954, 1.3924, 1.2756], "mean": 1.4080, "sd": 0.0873, "rank1": 20.0},
    "BiLSTM":       {"vals": [1.4998, 1.4563, 1.5102, 1.1857, 1.6364], "mean": 1.4577, "sd": 0.1661, "rank1": 20.0},
    "LSTM":         {"vals": [1.2099, 1.4833, 1.6415, 1.5563, 1.5196], "mean": 1.4821, "sd": 0.1631, "rank1": 20.0},
    "Ridge":        {"vals": [1.6033, 1.6033, 1.6033, 1.6033, 1.6033], "mean": 1.6033, "sd": 0.0000, "rank1": 0.0},
    "Mamba":        {"vals": [1.6356, 1.8272, 1.4736, 1.4772, 1.7134], "mean": 1.6254, "sd": 0.1530, "rank1": 0.0},
    "MLP":          {"vals": [1.7086, 1.6651, 2.2247, 1.6935, 1.8502], "mean": 1.8284, "sd": 0.2328, "rank1": 0.0},
}

seed42_vals = {k: v["vals"][0] for k, v in data.items()}
order = sorted(data.keys(), key=lambda k: data[k]["mean"])

fig, ax = plt.subplots(figsize=(12.5, 7.9), dpi=300)

x = np.arange(len(order))
jitter_rng = np.random.default_rng(0)

y_top = 2.55
ax.set_ylim(1.0, y_top)
ax.set_xlim(-0.6, len(order) - 1 + 0.6)

for i, arch in enumerate(order):
    vals = np.array(data[arch]["vals"])
    mean = data[arch]["mean"]
    sd = data[arch]["sd"]

    jitter = jitter_rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(x[i] + jitter, vals, s=26, color=K["grey"], alpha=0.75,
               zorder=3, edgecolor="none", label="Individual seed RMSE" if i == 0 else None)

    ax.errorbar(x[i], mean, yerr=sd, fmt="o", color=K["navy"], ecolor=K["navy"],
                elinewidth=1.6, capsize=5, markersize=8, zorder=5,
                label="Five-seed mean \u00b1 SD" if i == 0 else None)

    ax.scatter(x[i], seed42_vals[arch], s=95, marker="*", color=K["accent"],
               zorder=6, edgecolor="white", linewidth=0.6,
               label="Seed-42 value (preregistered single split)" if i == 0 else None)

    col_top = max(vals.max(), mean + sd, seed42_vals[arch])
    ax.text(x[i], col_top + 0.075, f"#1 in {data[arch]['rank1']:.0f}% of seeds",
            ha="center", va="bottom", fontsize=9.0, color=K["sub"])

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel("RMSE (log-ROAS, test set)", fontsize=11.5)

ax.axhline(data["BayesianLSTM"]["mean"], color=K["navy"], linewidth=0.6, linestyle=":", alpha=0.5, zorder=1)

ax.grid(axis="y", color=K["lgrey"], linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Reserve a generous header block and place title / subtitle / legend as
# independent figure-level elements with fixed, well-separated y positions
# (figure fraction) rather than relying on matplotlib's automatic title
# padding, which is what caused the title and subtitle to collide.
fig.subplots_adjust(top=0.76, bottom=0.19, left=0.08, right=0.97)

fig.text(0.045, 0.975,
          "Regression-stage forecasting: five-seed RMSE distribution vs. Seed-42 single split",
          ha="left", va="top", fontsize=14.5, fontweight="bold", color=K["text"])

fig.text(0.045, 0.928,
          "Sorted by five-seed mean RMSE; Bayesian LSTM is lowest-mean and most stable, not LSTM",
          ha="left", va="top", fontsize=11, style="italic", color=K["sub"])

legend_handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(legend_handles, legend_labels, loc="upper left",
           bbox_to_anchor=(0.045, 0.878), ncol=3, frameon=False, fontsize=10,
           handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0)

fig.text(0.5, 0.055,
          "Note: Seed-42 favors LSTM (red star, RMSE=1.2099), but across the five-seed panel LSTM ranks #1 only once (20%)\n"
          "and shows the second-largest cross-seed SD (0.1631). Bayesian LSTM has the lowest five-seed mean (1.3604)\n"
          "and smallest SD (0.0240), ranking #1 in 40% of seeds. Ridge has SD=0 because it is a closed-form fit,\n"
          "not a stochastic architecture.",
          ha="center", va="top", fontsize=8.8, color=K["sub"])

out_path = "/mnt/user-data/outputs/assets/fig4_regression_fiveseed.png"
plt.savefig(out_path, dpi=300, facecolor="white")
print("Saved:", out_path)
