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
