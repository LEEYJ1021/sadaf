"""
fig3_framework_architecture.py

Regenerates Figure 3 ("SADAF framework architecture") -- Section 5 of the
manuscript.

Fix applied: the previous version of this diagram's Explainability (Pillar
III) box still showed the pre-redesign, sequence-level H5 statistics
("Clusters n = 7, 8, 9" and "Rank corr. rho = .56-.83"), left over from
before H5 moved to row-level individual attribution (Individual SHAP +
Permutation SHAP + Integrated Gradients). This directly contradicted
Table 9/10 and Section 6.7 of the manuscript, which report row-level
cluster sizes of n = 1,214 / 217 / 28 (39/41/13 ad groups) and a
cross-method Spearman range of rho = .607-.964.

This version updates the Explainability box to state:
    - Clusters n = 1,214 / 217 / 28 (row-level)
    - H5 3-method attribution (Individual SHAP, Permutation SHAP,
      Integrated Gradients) -- all reject H0 (KW p<.0001/.0001/.0001... )
    - Rank corr. rho = .607-.964
All other boxes (I. Causal estimation, II. Bayesian prediction, bottom
robustness/scope band) are unchanged from the prior version, since they
already matched the manuscript text.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"]
mpl.rcParams["axes.unicode_minus"] = False

K = {
    "white":  "#FFFFFF",
    "l1":     "#F2F2F2",
    "l2":     "#E4E4E4",
    "l3":     "#D3D3D3",
    "border": "#1A1A1A",
    "text":   "#111111",
    "sub":    "#3A3A3A",
}

fig, ax = plt.subplots(figsize=(9.7, 9.5), dpi=300)
ax.set_xlim(0, 680)
ax.set_ylim(0, 650)
ax.invert_yaxis()
ax.axis("off")


def box(x, y, w, h, fill=K["white"], lw=1.1, dashed=False):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=6",
        linewidth=lw, edgecolor=K["border"], facecolor=fill, zorder=2,
        linestyle=(0, (4, 2)) if dashed else "solid",
    )
    ax.add_patch(b)


def arrow(x1, y1, x2, y2, lw=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                         linewidth=lw, color=K["border"], zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def line(x1, y1, x2, y2, lw=0.9):
    ax.plot([x1, x2], [y1, y2], color=K["border"], linewidth=lw, zorder=1)


def ctext(cx, cy, s, size=10.5, bold=False, color=K["text"], style="normal"):
    ax.text(cx, cy, s, fontsize=size, fontweight="bold" if bold else "normal",
            color=color, va="center", ha="center", fontstyle=style)


FS_TITLE, FS_SUB, FS_LIST = 12, 10, 9.6

# ---------------- Row 1: raw data ----------------
box(40, 40, 600, 78, fill=K["white"])
ctext(340, 58, "Raw hourly ad-group performance data", FS_TITLE, bold=True)
ctext(340, 78, "89,675 rows \u00d7 32 columns \u00b7 Korean search-advertising ecosystem \u00b7 March 2025",
      FS_SUB, color=K["sub"])
ctext(340, 96, "Data standardized and provided by SearchM Co., Ltd. \u2014 official agency unifying",
      FS_SUB - 0.6, color=K["sub"], style="italic")
ctext(340, 110, "Naver and Kakao search-advertising operations across a 37-advertiser panel",
      FS_SUB - 0.6, color=K["sub"], style="italic")

arrow(340, 118, 340, 138)

# ---------------- Row 2: ZINB diagnosis ----------------
box(40, 138, 600, 52, fill=K["l1"])
ctext(340, 156, "Structural diagnosis", FS_TITLE, bold=True)
ctext(340, 178, "ZINB preferred over ZIP (\u0394AIC = \u22122798.9); overdispersion confirmed", FS_SUB, color=K["sub"])

line(340, 190, 340, 200)
line(135, 200, 545, 200)
arrow(135, 200, 135, 210)
arrow(340, 200, 340, 210)
arrow(545, 200, 545, 210)

# ---------------- Row 3: pillar headers ----------------
headers = [
    (40,  K["white"], "I. Causal estimation"),
    (245, K["l1"],    "II. Bayesian prediction"),
    (450, K["l2"],    "III. Explainability"),
]
for x, fill, label in headers:
    box(x, 210, 190, 48, fill=fill, lw=1.3)
    cx = x + 95
    words = label.split(" ")
    mid = len(words) // 2 + (len(words) % 2)
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    ctext(cx, 226, line1, FS_TITLE, bold=True)
    ctext(cx, 244, line2, FS_TITLE, bold=True)
    arrow(cx, 258, cx, 268)

# ---------------- Row 4: pillar detail boxes ----------------
CHK = r"$\checkmark$"
DEP = r"$\triangle$"

detail_specs = [
    (40,  K["white"], [
        ("n = 14,987 matched pairs", False),
        (f"H1  CTR \u2192 conversion   {CHK}", True),
        (f"H2  Depth suppressor   {DEP}", True),
        (f"H3  Search \u00d7 Shopping   {CHK}", True),
    ]),
    (245, K["l1"], [
        ("174 \u2192 870 augmented sequences", False),
        (f"H4a  Classification   {CHK}", True),
        (f"H4b  Regression   {CHK}", True),
        (f"H4c  Overfit diagnostic   {CHK}", True),
        ("R1  seq-length check (Mamba)", False),
    ]),
    (450, K["l2"], [
        ("Clusters n = 1,214 / 217 / 28", False),
        ("(row-level; 39/41/13 ad groups)", False),
        (f"H5  3-method attribution   {CHK}", True),
        ("Ind.SHAP + Perm.SHAP + IG", False),
        ("Rank corr.  \u03c1 = .607\u2013.964", False),
    ]),
]
for x, fill, items in detail_specs:
    box(x, 268, 190, 156, fill=fill, lw=1.1)
    cx = x + 95
    y0 = 292
    for i, (txt, is_hyp) in enumerate(items):
        ctext(cx, y0 + i * 23, txt, FS_LIST, bold=is_hyp,
              color=K["text"] if is_hyp else K["sub"],
              style="normal" if is_hyp else "italic")

for cx in (135, 340, 545):
    line(cx, 424, cx, 434)
line(135, 434, 545, 434)
arrow(340, 434, 340, 444)

# ---------------- Row 5: integrated verdict ----------------
box(40, 444, 600, 46, fill=K["l1"], lw=1.3)
ctext(340, 467, "Integrated verdict \u2014 H1\u2013H5 evaluated (raw + FDR-corrected)", FS_TITLE, bold=True)

arrow(340, 490, 340, 510)

# ---------------- Row 6: robustness / scope ----------------
box(40, 510, 600, 46, fill=K["white"], lw=1.3, dashed=True)
ctext(340, 528, "Robustness \u0026 scope \u2014 R1 seq-length \u00b7 R2 campaign shift \u00b7 RQ6 LOAO-CV (37 advertisers)",
      FS_SUB - 1.2, color=K["text"])
ctext(340, 544, "within Korean multi-platform (Naver + Kakao) search-advertising scope only",
      FS_SUB - 1.2, color=K["text"])

# ---------------- Legend ----------------
legend_items = [
    (150, K["white"], "I  Causal estimation"),
    (330, K["l1"],    "II  Bayesian prediction"),
    (520, K["l2"],    "III  Explainability"),
]
ly = 584
for cx, fill, label in legend_items:
    sw_w = 16
    box(cx - sw_w - 6, ly - 7, sw_w, 14, fill=fill, lw=1.0)
    ax.text(cx + 6, ly, label, fontsize=FS_SUB, color=K["text"], va="center", ha="left")

ax.text(340, 604, f"{CHK}  supported as originally hypothesized      "
                   f"{DEP}  informative departure from the hypothesized mechanism (see \u00a76.2)",
        fontsize=8.3, color=K["text"], va="center", ha="center")

ax.text(340, 617, "Dashed border denotes robustness / scope-check content",
        fontsize=8.5, color=K["sub"], va="center", ha="center", fontstyle="italic")

plt.tight_layout(pad=0.4)
out_path = "/mnt/user-data/outputs/assets/fig3_framework_architecture.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out_path)
