import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

# ---------- top-journal style: grayscale, serif font ----------
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

fig, ax = plt.subplots(figsize=(9.7, 9.3), dpi=300)
ax.set_xlim(0, 680)
ax.set_ylim(0, 640)
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

FS_TITLE, FS_SUB, FS_LIST = 12, 10, 10.2

# ---------------- Row 1: raw data — reframed as SearchM multi-platform data ----------------
box(40, 40, 600, 76, fill=K["white"])
ctext(340, 58, "Raw hourly ad-group performance data", FS_TITLE, bold=True)
ctext(340, 78, "89,675 rows \u00d7 32 columns \u00b7 Korean search-advertising ecosystem \u00b7 March 2025",
      FS_SUB, color=K["sub"])
ctext(340, 96, "Data standardized and provided by SearchM Co., Ltd. \u2014 official agency unifying",
      FS_SUB - 0.6, color=K["sub"], style="italic")
ctext(340, 110, "Naver and Kakao search-advertising operations",
      FS_SUB - 0.6, color=K["sub"], style="italic")

arrow(340, 116, 340, 136)

# ---------------- Row 2: ZINB diagnosis ----------------
box(40, 136, 600, 52, fill=K["l1"])
ctext(340, 154, "Structural diagnosis", FS_TITLE, bold=True)
ctext(340, 176, "ZINB preferred over ZIP (\u0394AIC = \u22122798.9); overdispersion confirmed", FS_SUB, color=K["sub"])

# fan-out to 3 headers
line(340, 188, 340, 198)
line(135, 198, 545, 198)
arrow(135, 198, 135, 208)
arrow(340, 198, 340, 208)
arrow(545, 198, 545, 208)

# ---------------- Row 3: pillar headers ----------------
headers = [
    (40,  K["white"], "I. Causal estimation"),
    (245, K["l1"],    "II. Bayesian prediction"),
    (450, K["l2"],    "III. Explainability"),
]
for x, fill, label in headers:
    box(x, 208, 190, 48, fill=fill, lw=1.3)
    cx = x + 95
    words = label.split(" ")
    mid = len(words) // 2 + (len(words) % 2)
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    ctext(cx, 224, line1, FS_TITLE, bold=True)
    ctext(cx, 242, line2, FS_TITLE, bold=True)
    arrow(cx, 256, cx, 266)

# ---------------- Row 4: pillar detail boxes ----------------
CHK = r"$\checkmark$"
DEP = r"$\triangle$"   # informative departure from the hypothesized mechanism (not a binary support/non-support)
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
        ("Clusters n = 7, 8, 9", False),
        (f"H5  SHAP attribution   {CHK}", True),
        ("Rank corr.  \u03c1 = .56\u2013.83", False),
    ]),
]
for x, fill, items in detail_specs:
    box(x, 266, 190, 152, fill=fill, lw=1.1)
    cx = x + 95
    y0 = 292
    for i, (txt, is_hyp) in enumerate(items):
        ctext(cx, y0 + i * 26, txt, FS_LIST, bold=is_hyp,
              color=K["text"] if is_hyp else K["sub"],
              style="normal" if is_hyp else "italic")

# converge to verdict
for cx in (135, 340, 545):
    line(cx, 418, cx, 428)
line(135, 428, 545, 428)
arrow(340, 428, 340, 438)

# ---------------- Row 5: integrated verdict ----------------
box(40, 438, 600, 46, fill=K["l1"], lw=1.3)
ctext(340, 461, "Integrated verdict \u2014 H1\u2013H5 evaluated (raw + FDR-corrected)", FS_TITLE, bold=True)

arrow(340, 484, 340, 504)

# ---------------- Row 6: robustness / scope ----------------
box(40, 504, 600, 46, fill=K["white"], lw=1.3, dashed=True)
ctext(340, 522, "Robustness \u0026 scope \u2014 R1 seq-length \u00b7 R2 campaign shift \u00b7 RQ6 LOAO-CV (37 advertisers)",
      FS_SUB - 1.2, color=K["text"])
ctext(340, 538, "within Korean multi-platform (Naver + Kakao) search-advertising scope only",
      FS_SUB - 1.2, color=K["text"])

# ---------------- Legend ----------------
legend_items = [
    (150, K["white"], "I  Causal estimation"),
    (330, K["l1"],    "II  Bayesian prediction"),
    (520, K["l2"],    "III  Explainability"),
]
ly = 578
for cx, fill, label in legend_items:
    sw_w = 16
    box(cx - sw_w - 6, ly - 7, sw_w, 14, fill=fill, lw=1.0)
    ax.text(cx + 6, ly, label, fontsize=FS_SUB, color=K["text"], va="center", ha="left")

ax.text(340, 598, f"{CHK}  supported as originally hypothesized      "
                   f"{DEP}  informative departure from the hypothesized mechanism (see \u00a76.2)",
        fontsize=8.3, color=K["text"], va="center", ha="center")

ax.text(340, 611, "Dashed border denotes robustness / scope-check content",
        fontsize=8.5, color=K["sub"], va="center", ha="center", fontstyle="italic")

plt.tight_layout(pad=0.4)
out_path = "/mnt/user-data/outputs/sadaf_framework_architecture_v4.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", out_path)
