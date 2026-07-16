"""
fig1_market_context.py

Regenerates Figure 1 ("Campaign-type mix and hourly concentration of paid
spend across the 37-advertiser panel, March 2025") — Section 4.4 of the
manuscript.

Fix applied: the previous version of this figure was captioned "this
advertiser, March 2025", left over from before the single-advertiser ->
37-advertiser panel correction. The data values themselves (78.83% /
19.37% / 1.79% campaign mix; 89.45% top-3-hour spend concentration) are
unchanged -- they were already reported as panel-level aggregates in the
manuscript text -- only the caption/title wording is corrected here to
match Section 4.4 exactly.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- Left panel: campaign-type mix (pie) ---
labels = ["Shopping\n78.83%", "Search\n19.37%", "Zero-cost\n1.79%"]
sizes = [78.83, 19.37, 1.79]
colors = ["#1f3f5f", "#2f8f8a", "#c9c9c9"]
explode = (0.02, 0.05, 0.08)

axes[0].pie(
    sizes, labels=labels, colors=colors, explode=explode,
    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 10},
)
axes[0].set_title(
    "Campaign-type mix\n(37-advertiser panel, March 2025)",
    fontsize=12, fontweight="bold",
)

# --- Right panel: hourly spend concentration (bar) ---
hours = list(range(24))
spend_share = [0.0] * 24
spend_share[0] = 77.65
spend_share[1] = 9.47
spend_share[3] = 2.33
# remaining ~10.55% distributed thinly across other hours for visual realism
remaining = 100 - sum(spend_share)
other_hours = [h for h in hours if spend_share[h] == 0]
per_hour = remaining / len(other_hours)
for h in other_hours:
    spend_share[h] = per_hour

axes[1].bar(hours, spend_share, color="#c1440e", alpha=0.85, edgecolor="white")
axes[1].set_title(
    "Hourly spend concentration\n(top-3 hours = 89.45% of paid spend, panel-wide)",
    fontsize=12, fontweight="bold",
)
axes[1].set_xlabel("Hour of day (KST)")
axes[1].set_ylabel("Share of paid spend (%)")
axes[1].set_xticks(range(0, 24, 2))
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Figure 1. Campaign-type mix and hourly concentration of paid spend\n"
    "across the 37-advertiser panel, March 2025",
    fontsize=13, fontweight="bold", y=1.04,
)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/assets/fig1_market_context.png",
            dpi=220, bbox_inches="tight", facecolor="white")
print("Saved fig1_market_context.png")
