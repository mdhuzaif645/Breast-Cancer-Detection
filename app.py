"""Breast Cancer Prediction Studio - Logistic Regression + Gradio (Hugging Face Space).
 
Tabs
  1. Diagnose      : sliders -> verdict, breast anatomy diagram with a magnified cell nucleus that
                     changes with the inputs, risk gauge, feature effects, what-if curves, distributions
  2. Decision map  : the logistic-regression boundary drawn for any two features
  3. How it works  : model, weights, limitations
"""
import warnings
 
warnings.filterwarnings("ignore")
 
from pathlib import Path
 
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from scipy.stats import gaussian_kde
 
# ZeroGPU Spaces need one @spaces.GPU function to start. It is never called (no GPU is used).
# The try/except keeps the app working on CPU hardware and on your own computer.
try:
    import spaces
 
    @spaces.GPU
    def _gpu_placeholder():
        return None
 
except ImportError:
    pass
 
# ------------------------------------------------------------------------------ setup
BASE = Path(__file__).parent
scaler, model = joblib.load(BASE / "easy_model.joblib")
 
COLS = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"]
PRETTY = {
    "mean_radius": "Mean radius",
    "mean_texture": "Mean texture",
    "mean_perimeter": "Mean perimeter",
    "mean_area": "Mean area",
    "mean_smoothness": "Mean smoothness",
}
# min, max, median of the training data
RANGES = {
    "mean_radius": (6.981, 28.11, 13.37),
    "mean_texture": (9.71, 39.28, 18.84),
    "mean_perimeter": (43.79, 188.5, 86.24),
    "mean_area": (143.5, 2501.0, 551.1),
    "mean_smoothness": (0.0526, 0.1634, 0.0959),
}
MAL, BEN = "#d6336c", "#2f80c0"
BANDS = [(0.25, "Low", "#2e9e5b"), (0.50, "Borderline", "#e0b000"),
         (0.75, "Elevated", "#ee7b1a"), (1.01, "High", "#d32f2f")]
 
try:
    data = pd.read_csv(BASE / "Breast_cancer_data.csv")
    MEAN_MAL = data[data.diagnosis == 0][COLS].mean().tolist()
    MEAN_BEN = data[data.diagnosis == 1][COLS].mean().tolist()
except Exception:  # the app still works without the CSV, only the data plots are skipped
    data = None
    MEAN_MAL = [17.46, 21.60, 115.37, 978.38, 0.1029]
    MEAN_BEN = [12.15, 17.91, 78.08, 462.79, 0.0925]
MEDIAN = [RANGES[c][2] for c in COLS]
 
 
def risk_band(p):
    for upper, label, colour in BANDS:
        if p < upper:
            return label, colour
    return BANDS[-1][1], BANDS[-1][2]
 
 
def p_malignant(df):
    """Probability of malignant (class 0) for every row of df."""
    return 1.0 - model.predict_proba(scaler.transform(df[COLS]))[:, 1]
 
 
def make_row(vals):
    return pd.DataFrame([list(vals)], columns=COLS)
 
 
def empty_fig(text):
    fig = Figure(figsize=(6, 2), facecolor="white")
    ax = fig.subplots()
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#666")
    ax.axis("off")
    return fig
 
 
# ------------------------------------------------------------- anatomy + nucleus diagram
def ribbon_svg(size=60, colour="#e75480"):
    return f"""<svg width="{size}" height="{size * 1.15:.0f}" viewBox="0 0 100 115" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="{colour}" stroke-width="13" stroke-linecap="round" stroke-linejoin="round">
        <path d="M50 12 C 18 12, 18 55, 50 70 C 80 85, 90 100, 82 108"/>
        <path d="M50 12 C 82 12, 82 55, 50 70 C 20 85, 10 100, 18 108"/></g></svg>"""
 
 
def nucleus_svg(radius, texture, smooth, colour, cx=600, cy=205, panel_r=135):
    """Schematic magnified nucleus: size ~ radius, edge roughness ~ smoothness, speckles ~ texture."""
    rng = np.random.RandomState(7)
    scale = 4.2
    R = radius * scale
    rough = float(np.clip((smooth - 0.05) / 0.11, 0, 1))
    amp = 0.015 + 0.11 * rough
    ks = np.arange(2, 9)
    phases = rng.uniform(0, 2 * np.pi, len(ks))
    weights = rng.uniform(0.4, 1.0, len(ks))
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    wobble = sum(w * np.sin(k * theta + ph) for k, ph, w in zip(ks, phases, weights)) / np.sqrt(len(ks) / 2)
    rr = R * (1 + amp * wobble)
    pts = " ".join(f"{cx + r * np.cos(t):.1f},{cy + r * np.sin(t):.1f}" for r, t in zip(rr, theta))
 
    n_dots = int(20 + (np.clip(texture, 9.7, 39.3) - 9.7) / (39.3 - 9.7) * 140)
    dots = []
    while len(dots) < n_dots:
        x, y = rng.uniform(-1, 1, 2)
        if x * x + y * y < 0.78:
            size = rng.uniform(1.2, 2.2 + 1.6 * texture / 40)
            dots.append(f"<circle cx='{cx + x * R:.1f}' cy='{cy + y * R:.1f}' r='{size:.1f}' fill='#3f2a78' opacity='0.55'/>")
 
    ref_r = 12.15 * scale  # typical benign nucleus, drawn as a dashed scale ring
    smear = "".join(
        f"<circle cx='{cx + rng.uniform(-115, 115):.0f}' cy='{cy + rng.uniform(-115, 115):.0f}' r='{rng.uniform(5, 9):.1f}' fill='#f6c9d6' opacity='0.55'/>"
        for _ in range(14))
    return f"""
    <defs><clipPath id="zoomclip"><circle cx="{cx}" cy="{cy}" r="{panel_r}"/></clipPath></defs>
    <circle cx="{cx}" cy="{cy}" r="{panel_r}" fill="#fdf1f5"/>
    <g clip-path="url(#zoomclip)">
      {smear}
      <circle cx="{cx}" cy="{cy}" r="{ref_r:.1f}" fill="none" stroke="#7a7a7a" stroke-width="1.6" stroke-dasharray="5 5"/>
      <polygon points="{pts}" fill="#b9a4e6" fill-opacity="0.85" stroke="{colour}" stroke-width="3.5" stroke-linejoin="round"/>
      {''.join(dots)}
    </g>
    <circle cx="{cx}" cy="{cy}" r="{panel_r}" fill="none" stroke="{colour}" stroke-width="5"/>
    <text x="{cx}" y="{cy + panel_r + 22}" text-anchor="middle" font-size="12.5" fill="#444">radius {radius:.1f}  |  texture {texture:.1f}  |  smoothness {smooth:.3f}</text>
    <text x="{cx}" y="{cy + panel_r + 40}" text-anchor="middle" font-size="11" fill="#777">dashed ring = typical benign nucleus size</text>
    """
 
 
def anatomy_html(p_mal, radius, texture, smooth):
    band, colour = risk_band(p_mal)
 
    def badge(n, x, y):
        return (f"<circle cx='{x}' cy='{y}' r='11' fill='#8e1b4c'/>"
                f"<text x='{x}' y='{y + 4.5}' text-anchor='middle' font-size='12' font-weight='700' fill='#fff'>{n}</text>")
 
    # lobule clusters (milk glands) and the ducts that lead to the nipple
    clusters = [(215, 135), (255, 225), (215, 315), (300, 175)]
    lobules, ducts = [], []
    rng = np.random.RandomState(3)
    for (x, y) in clusters:
        for _ in range(7):
            dx, dy = rng.uniform(-22, 22, 2)
            lobules.append(f"<circle cx='{x + dx:.0f}' cy='{y + dy:.0f}' r='{rng.uniform(6, 9):.1f}' fill='#f6a9c4' stroke='#d6336c' stroke-width='1.4'/>")
        ducts.append(f"<path d='M{x} {y} C {x + 50} {y}, {x + 70} 222, 372 222' fill='none' stroke='#c2185b' stroke-width='3' stroke-linecap='round'/>")
    fat = "".join(
        f"<ellipse cx='{rng.uniform(150, 340):.0f}' cy='{rng.uniform(100, 340):.0f}' rx='{rng.uniform(8, 16):.0f}' ry='{rng.uniform(5, 9):.0f}' fill='#fff3e2' opacity='0.8'/>"
        for _ in range(16))
 
    return f"""
<div class="bc-diagram">
<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif" role="img"
     aria-label="Schematic breast anatomy with a magnified cell nucleus">
  <text x="20" y="17" font-size="15" font-weight="700" fill="#8e1b4c">Breast anatomy (schematic side view)</text>
  <text x="600" y="17" text-anchor="middle" font-size="15" font-weight="700" fill="#8e1b4c">Cell nucleus, magnified (schematic)</text>
 
  <!-- chest wall: ribs and muscle -->
  <rect x="38" y="45" width="42" height="335" rx="10" fill="#ecdcc4"/>
  <ellipse cx="59" cy="90" rx="15" ry="26" fill="#d8c0a0" stroke="#b89a76"/>
  <ellipse cx="59" cy="170" rx="15" ry="26" fill="#d8c0a0" stroke="#b89a76"/>
  <ellipse cx="59" cy="250" rx="15" ry="26" fill="#d8c0a0" stroke="#b89a76"/>
  <ellipse cx="59" cy="330" rx="15" ry="26" fill="#d8c0a0" stroke="#b89a76"/>
  <rect x="80" y="55" width="30" height="315" rx="12" fill="#d98a80" stroke="#b96a60"/>
 
  <!-- breast tissue -->
  <path d="M110 75 C 200 58, 330 128, 372 203 L 384 208 C 398 214, 398 230, 384 236 L 372 241 C 330 312, 200 352, 110 345 Z"
        fill="#fbe1c6" stroke="#d9a679" stroke-width="3"/>
  {fat}
  <ellipse cx="376" cy="222" rx="9" ry="30" fill="#efbfa8"/>
  <ellipse cx="392" cy="222" rx="10" ry="14" fill="#dc9a86"/>
  {''.join(ducts)}
  {''.join(lobules)}
 
  <!-- lymph nodes -->
  <ellipse cx="176" cy="60" rx="11" ry="7" fill="#9ccc9c" stroke="#5b8f5b"/>
  <ellipse cx="208" cy="62" rx="9" ry="6" fill="#9ccc9c" stroke="#5b8f5b"/>
 
  <!-- sampled area: pulsing ring + zoom lines -->
  <circle cx="255" cy="225" r="36" fill="none" stroke="{colour}" stroke-width="3.5" stroke-dasharray="6 4">
    <animate attributeName="r" values="32;40;32" dur="2.2s" repeatCount="indefinite"/></circle>
  <line x1="280" y1="196" x2="492" y2="120" stroke="#999" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="280" y1="256" x2="492" y2="292" stroke="#999" stroke-width="1.4" stroke-dasharray="4 4"/>
 
  {badge(1, 59, 40)}{badge(2, 95, 40)}{badge(3, 175, 90)}{badge(4, 190, 165)}
  {badge(5, 330, 190)}{badge(6, 376, 262)}{badge(7, 192, 42)}
 
  {nucleus_svg(radius, texture, smooth, colour)}
</svg>
<div class="bc-legend">
  <span><b>1</b> Chest wall &amp; ribs</span><span><b>2</b> Pectoral muscle</span><span><b>3</b> Fat tissue</span>
  <span><b>4</b> Lobules (milk glands)</span><span><b>5</b> Milk ducts</span>
  <span><b>6</b> Nipple &amp; areola</span><span><b>7</b> Lymph nodes</span>
</div>
<div class="bc-note">The pulsing ring marks where cells are sampled. The magnified nucleus is redrawn from your sliders:
<b>size</b> follows mean radius, <b>edge roughness</b> follows mean smoothness, <b>dark speckles</b> follow mean texture.
Schematic illustration for learning, not a medical image.</div>
</div>"""
 
 
# ------------------------------------------------------------------------- charts
def gauge_fig(p_mal, thr):
    fig = Figure(figsize=(5.4, 3.3), facecolor="white")
    ax = fig.subplots()
    lo = 0.0
    for upper, _, colour in BANDS:
        hi = min(upper, 1.0)
        ax.add_patch(Wedge((0, 0), 1.0, 180 - hi * 180, 180 - lo * 180, width=0.30,
                           facecolor=colour, edgecolor="white", lw=2, alpha=0.92))
        lo = hi
    a = np.deg2rad(180 - p_mal * 180)
    ax.plot([0, 0.80 * np.cos(a)], [0, 0.80 * np.sin(a)], color="#222", lw=3.5, solid_capstyle="round")
    ax.add_patch(Wedge((0, 0), 0.07, 0, 360, facecolor="#222"))
    t = np.deg2rad(180 - thr * 180)
    ax.plot([0.66 * np.cos(t), 1.10 * np.cos(t)], [0.66 * np.sin(t), 1.10 * np.sin(t)],
            color="#222", lw=1.6, ls=(0, (3, 2)))
    ax.text(1.14 * np.cos(t), 1.14 * np.sin(t) + 0.05, f"threshold {thr:.2f}", ha="center", fontsize=8, color="#444")
    ax.text(0, -0.30, f"{p_mal:.1%}", ha="center", fontsize=26, fontweight="bold", color="#222")
    ax.text(0, -0.52, "probability of malignant", ha="center", fontsize=10, color="#666")
    ax.text(-1.15, -0.06, "0%", ha="center", fontsize=8, color="#888")
    ax.text(1.15, -0.06, "100%", ha="center", fontsize=8, color="#888")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.65, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
 
 
def contributions(X):
    z = scaler.transform(X[COLS])[0]
    return z, -model.coef_[0] * z  # log-odds pushed towards MALIGNANT by each feature
 
 
def contribution_fig(X):
    z, contrib = contributions(X)
    order = np.argsort(np.abs(contrib))
    labels = [f"{PRETTY[COLS[i]]}\n{X.iloc[0][COLS[i]]:.4g}  (z = {z[i]:+.1f})" for i in order]
    vals = contrib[order]
    fig = Figure(figsize=(6.4, 3.8), facecolor="white")
    ax = fig.subplots()
    ax.barh(labels, vals, color=[MAL if v > 0 else BEN for v in vals], height=0.6)
    ax.axvline(0, color="#333", lw=1)
    for y, v in enumerate(vals):
        ax.text(v + (0.05 if v >= 0 else -0.05), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color="#333")
    span = max(1.0, np.abs(vals).max() * 1.35)
    ax.set_xlim(-span, span)
    ax.set_xlabel("<- pushes towards benign | pushes towards malignant ->", fontsize=9)
    ax.set_title("Why this prediction?", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
 
 
def whatif_fig(X, thr):
    fig = Figure(figsize=(15, 3.1), facecolor="white")
    axes = fig.subplots(1, 5, sharey=True)
    base = X.iloc[0]
    p_now = float(p_malignant(X)[0])
    for ax, c in zip(axes, COLS):
        lo, hi, _ = RANGES[c]
        grid = np.linspace(lo * 0.92, hi * 1.04, 120)
        sweep = pd.DataFrame([base.values] * len(grid), columns=COLS)
        sweep[c] = grid
        p = p_malignant(sweep)
        ax.fill_between(grid, p, color=MAL, alpha=0.16)
        ax.plot(grid, p, color=MAL, lw=2.2)
        ax.axhline(thr, color="#555", ls="--", lw=1)
        ax.scatter([base[c]], [p_now], s=70, color="#111", zorder=5)
        ax.set_title(PRETTY[c], fontsize=10)
        ax.set_ylim(0, 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(malignant)")
    fig.suptitle("What-if: change ONE measurement (others fixed). The black dot is the current case.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
 
 
def distribution_fig(X):
    if data is None:
        return empty_fig("Upload Breast_cancer_data.csv to the Space to see the distribution plots.")
    fig = Figure(figsize=(11, 5.4), facecolor="white")
    axes = fig.subplots(2, 3).ravel()
    for ax, c in zip(axes, COLS):
        xs = np.linspace(data[c].min() * 0.9, data[c].max() * 1.05, 250)
        for cls, colour in ((0, MAL), (1, BEN)):
            ys = gaussian_kde(data[data.diagnosis == cls][c])(xs)
            ax.fill_between(xs, ys, color=colour, alpha=0.30)
            ax.plot(xs, ys, color=colour, lw=1.6)
        ax.axvline(X.iloc[0][c], color="#111", ls="--", lw=2)
        ax.set_title(PRETTY[c], fontsize=10)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    last = axes[-1]
    last.axis("off")
    last.plot([], [], color=MAL, lw=6, alpha=0.5, label="Malignant (training data)")
    last.plot([], [], color=BEN, lw=6, alpha=0.5, label="Benign (training data)")
    last.plot([], [], color="#111", ls="--", lw=2, label="This case")
    last.legend(loc="center", frameon=False, fontsize=10)
    fig.suptitle("Where does this case sit compared with real patients?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
 
 
def result_card(p_mal, thr, X):
    flagged = p_mal >= thr
    band, colour = risk_band(p_mal)
    _, contrib = contributions(X)
    top = int(np.argmax(np.abs(contrib)))
    direction = "towards malignant" if contrib[top] > 0 else "towards benign"
    vcol = MAL if flagged else BEN
    verdict = "Malignant pattern" if flagged else "Benign pattern"
    conf = p_mal if flagged else 1 - p_mal
    out = [PRETTY[c] for c in COLS if not RANGES[c][0] <= X.iloc[0][c] <= RANGES[c][1]]
    warn = (f"<div class='bc-warn'>Outside the training range: {', '.join(out)}. Treat this result with extra caution.</div>"
            if out else "")
    return f"""
<div class="bc-card" style="border-left:8px solid {vcol}">
  <div class="bc-small">Model verdict at threshold {thr:.2f}</div>
  <div class="bc-verdict" style="color:{vcol}">{verdict}</div>
  <div class="bc-row">
    <span class="bc-pill" style="background:{colour}">Risk: {band}</span>
    <span>P(malignant) <b>{p_mal:.1%}</b></span>
    <span>P(benign) <b>{1 - p_mal:.1%}</b></span>
    <span>Confidence <b>{conf:.1%}</b></span>
  </div>
  <div class="bc-driver">Biggest driver: <b>{PRETTY[COLS[top]]}</b> pushes the result {direction}.</div>
  {warn}
  <div class="bc-note">Educational demo. Not a medical device and not a substitute for a doctor's diagnosis.</div>
</div>"""
 
 
# ------------------------------------------------------------------------ callbacks
def analyze(radius, texture, perimeter, area, smooth, thr):
    vals = [radius, texture, perimeter, area, smooth]
    if any(v is None for v in vals):
        raise gr.Error("Please fill in all five measurements.")
    X = make_row(vals)
    p = float(p_malignant(X)[0])
    return (result_card(p, thr, X), anatomy_html(p, radius, texture, smooth),
            gauge_fig(p, thr), contribution_fig(X), whatif_fig(X, thr), distribution_fig(X))
 
 
def decision_map(xf, yf, radius, texture, perimeter, area, smooth, thr):
    if xf == yf:
        return empty_fig("Pick two different features.")
    X = make_row([radius, texture, perimeter, area, smooth])
    inv = {v: k for k, v in PRETTY.items()}
    xc, yc = inv[xf], inv[yf]
    gx = np.linspace(RANGES[xc][0] * 0.9, RANGES[xc][1] * 1.05, 90)
    gy = np.linspace(RANGES[yc][0] * 0.9, RANGES[yc][1] * 1.05, 90)
    XX, YY = np.meshgrid(gx, gy)
    grid = pd.DataFrame([X.iloc[0].values] * XX.size, columns=COLS)
    grid[xc], grid[yc] = XX.ravel(), YY.ravel()
    P = p_malignant(grid).reshape(XX.shape)
 
    fig = Figure(figsize=(7.6, 5.6), facecolor="white")
    ax = fig.subplots()
    cmap = LinearSegmentedColormap.from_list("bm", [BEN, "#ffffff", MAL])
    cs = ax.contourf(XX, YY, P, levels=np.linspace(0, 1, 21), cmap=cmap, alpha=0.8)
    ax.contour(XX, YY, P, levels=[thr], colors="#111", linewidths=2)
    if data is not None:
        col = data.diagnosis.map({0: MAL, 1: BEN})
        ax.scatter(data[xc], data[yc], c=col, s=12, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.scatter([X.iloc[0][xc]], [X.iloc[0][yc]], marker="*", s=380, color="#ffd400", edgecolor="#111", linewidth=1.5, zorder=6)
    ax.set_xlabel(xf)
    ax.set_ylabel(yf)
    ax.set_title("Decision map: black line = where P(malignant) equals the threshold\n"
                 "pink side = malignant, blue side = benign, star = your case", fontsize=10)
    fig.colorbar(cs, ax=ax, label="P(malignant)")
    fig.tight_layout()
    return fig
 
 
def preset(kind):
    return {"malignant": MEAN_MAL, "benign": MEAN_BEN, "median": MEDIAN}[kind]
 
 
# ---------------------------------------------------------------------------- UI
CSS = """
<style>
.bc-header{display:flex;align-items:center;gap:18px;padding:18px 24px;border-radius:16px;
  background:linear-gradient(115deg,#fde0ec 0%,#fff7fa 55%,#e6f1fb 100%);border:1px solid #f3c2d5}
.bc-header h1{margin:0;font-size:28px;color:#8e1b4c;letter-spacing:.2px}
.bc-header p{margin:4px 0 0;color:#5a5a5a;font-size:14.5px}
.bc-card{padding:16px 20px;border-radius:14px;background:#fff;border:1px solid #e6e6e6;color:#222;
  box-shadow:0 2px 10px rgba(142,27,76,.08)}
.bc-verdict{font-size:30px;font-weight:800;margin:2px 0 10px}
.bc-small{font-size:12px;color:#777;text-transform:uppercase;letter-spacing:.07em}
.bc-row{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:15px;color:#222}
.bc-pill{color:#fff;padding:3px 13px;border-radius:999px;font-weight:700}
.bc-driver{margin-top:10px;font-size:14px;color:#333}
.bc-warn{margin-top:10px;padding:8px 12px;border-radius:8px;background:#fff4e0;color:#8a5200;font-size:13px}
.bc-note{margin-top:10px;font-size:12px;color:#777}
.bc-diagram{margin-top:12px;padding:12px 14px;border-radius:14px;background:#fff;border:1px solid #e6e6e6;
  box-shadow:0 2px 10px rgba(142,27,76,.08)}
.bc-diagram svg{width:100%;height:auto;display:block}
.bc-legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:6px;font-size:12.5px;color:#444}
.bc-legend b{display:inline-block;width:19px;height:19px;line-height:19px;text-align:center;border-radius:50%;
  background:#8e1b4c;color:#fff;font-size:11.5px;margin-right:4px}
.bc-foot{text-align:center;color:#888;font-size:12.5px;margin-top:8px}
</style>"""
 
HEADER = f"""{CSS}
<div class="bc-header">{ribbon_svg(60)}
  <div><h1>Breast Cancer Prediction Studio</h1>
  <p>Move the sliders: the anatomy diagram, the magnified cell and every graph update.
  Logistic Regression predicts <b>malignant</b> or <b>benign</b> and explains why.</p></div>
</div>"""
 
 
def slider(c):
    lo, hi, med = RANGES[c]
    dec = 4 if c == "mean_smoothness" else 2
    step = {"mean_smoothness": 0.0001, "mean_area": 0.1}.get(c, 0.01)
    return gr.Slider(round(lo * 0.9, dec), round(hi * 1.1, dec), value=round(med, dec), step=step, label=PRETTY[c])
 
 
with gr.Blocks(title="Breast Cancer Prediction Studio") as demo:
    gr.HTML(HEADER)
 
    with gr.Tabs():
        # ------------------------------------------------------------ diagnose
        with gr.Tab("Diagnose"):
            with gr.Row():
                with gr.Column(scale=4, min_width=320):
                    gr.Markdown("### 1. Enter the measurements")
                    sliders = [slider(c) for c in COLS]
                    thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision threshold (malignant if P >= threshold)",
                                    info="Lower = catch more cancers but more false alarms.")
                    go = gr.Button("Analyze", variant="primary")
                    gr.Markdown("**Quick examples**")
                    with gr.Row():
                        b_mal = gr.Button("Typical malignant", size="sm")
                        b_ben = gr.Button("Typical benign", size="sm")
                        b_med = gr.Button("Median", size="sm")
                with gr.Column(scale=7, min_width=420):
                    card = gr.HTML()
                    diagram = gr.HTML()
            with gr.Row():
                gauge = gr.Plot(show_label=False)
                contrib = gr.Plot(show_label=False)
            whatif = gr.Plot(show_label=False)
            dist = gr.Plot(show_label=False)
 
            ins = sliders + [thr]
            outs = [card, diagram, gauge, contrib, whatif, dist]
            go.click(analyze, ins, outs)
            for btn, kind in ((b_mal, "malignant"), (b_ben, "benign"), (b_med, "median")):
                btn.click(lambda k=kind: preset(k), None, sliders).then(analyze, ins, outs)
            demo.load(analyze, ins, outs)
 
        # -------------------------------------------------------- decision map
        with gr.Tab("Decision map"):
            gr.Markdown("Logistic Regression separates the classes with a **straight line** (in this 2-feature slice). "
                        "Choose two features. The other three stay at your slider values from the Diagnose tab.")
            with gr.Row():
                xd = gr.Dropdown(list(PRETTY.values()), value="Mean radius", label="X axis")
                yd = gr.Dropdown(list(PRETTY.values()), value="Mean texture", label="Y axis")
                mbtn = gr.Button("Draw map", variant="primary")
            mplot = gr.Plot(show_label=False)
            mbtn.click(decision_map, [xd, yd] + ins, mplot)
            demo.load(decision_map, [xd, yd] + ins, mplot)
 
        # ---------------------------------------------------------- how it works
        with gr.Tab("How it works"):
            weights = pd.DataFrame({"Feature": [PRETTY[c] for c in COLS],
                                    "Weight (scaled)": np.round(model.coef_[0], 2)})
            with gr.Row():
                gr.HTML(f"<div style='text-align:center;padding:12px'>{ribbon_svg(120)}"
                        "<div style='color:#8e1b4c;font-weight:600;margin-top:6px'>Early detection saves lives</div></div>")
                gr.Markdown("""
### The model
1. **Scale** each measurement (mean 0, standard deviation 1).
2. **Weighted sum:** `z = b0 + b1*x1 + ... + b5*x5`
3. **Sigmoid:** `P(benign) = 1 / (1 + e^(-z))`, and `P(malignant) = 1 - P(benign)`.
 
A **negative weight** means a higher value makes *malignant* more likely.
Radius, perimeter and area all measure size, so their weights share one effect.
 
### Limits
- Trained on 569 public records with only 5 summary features.
- The magnified nucleus and the anatomy drawing are **schematic**, made to help understanding.
- **Not a medical device.** Real diagnosis needs a doctor, imaging and biopsy.
""")
            gr.Dataframe(value=weights, interactive=False, label="Learned weights")
 
    gr.HTML("<div class='bc-foot'>Built with scikit-learn and Gradio. Educational project.</div>")
 
if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
 