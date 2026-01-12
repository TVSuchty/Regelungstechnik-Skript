#!/usr/bin/env python3
"""Reproduce Abbildung 7 (Nyquist) inkl. Zoom um (-1,0)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(__file__)
T = np.array([1.0, 0.2, 0.05])

def G0(w, K=1.0):
    s = 1j*w
    return K / np.prod(1 + T*s)

w = np.logspace(-3, 3, 6000)
G_base = np.array([G0(wi, 1.0) for wi in w])
im = np.imag(G_base)

# find Kcrit from Im(G)=0 and Re(G)<0
idx = np.where(np.diff(np.sign(im)) != 0)[0]
candidates = []
for i in idx:
    w1, w2 = w[i], w[i+1]
    im1, im2 = im[i], im[i+1]
    wr = w1 + (0 - im1) * (w2 - w1) / (im2 - im1)
    Gr = G0(wr, 1.0)
    if np.real(Gr) < 0:
        Kcrit = -1/np.real(Gr)
        candidates.append((wr, Gr, Kcrit))
w_star, G_star, Kcrit = sorted(candidates, key=lambda x: abs(x[2]-31.5))[0]

labels = ["stabil (K=20)", "grenzstabil (K≈%.2f)" % Kcrit, "instabil (K=40)"]
Ks = [20.0, float(Kcrit), 40.0]
K_values = dict(zip(labels, Ks))

# save CSV
for label, K in K_values.items():
    G = np.array([G0(wi, K) for wi in w])
    df = pd.DataFrame({"omega_rad_s": w, "Re": np.real(G), "Im": np.imag(G)})
    safe = label.split("(")[0].strip().replace(" ", "_").replace("≈", "_")
    fname = "nyquist_fig7_%s_K_%.4g.csv" % (safe, K)
    df.to_csv(os.path.join(OUT_DIR, fname), index=False)

# stable clearance
G_stable = np.array([G0(wi, 20.0) for wi in w])
dist = np.abs(G_stable - (-1+0j))
min_idx = int(np.argmin(dist))
min_dist = float(dist[min_idx])
w_min = float(w[min_idx])
G_min = G_stable[min_idx]

def add_arrows(ax, x, y, n_arrows=3):
    idxs = np.unique(np.round(np.linspace(900, len(x)-1100, n_arrows)).astype(int))
    for i in idxs:
        ax.annotate("", xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", lw=1))

xlim_zoom = (-1.6, 0.25)
ylim_zoom = (-1.2, 1.2)

def plot(style="modern", zoom=False, out_prefix="modern"):
    if style == "latex":
        plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm"})
    fig, ax = plt.subplots(figsize=(7.2,5.6), dpi=240)

    if style == "latex":
        styles = {
            labels[0]: dict(color="black", linestyle="-", linewidth=2),
            labels[1]: dict(color="black", linestyle="--", linewidth=2),
            labels[2]: dict(color="black", linestyle=":", linewidth=2.6),
        }
    else:
        styles = {lab: dict(linewidth=2) for lab in labels}

    for lab in labels:
        K = K_values[lab]
        G = np.array([G0(wi, K) for wi in w])
        ax.plot(np.real(G), np.imag(G), label=lab, **styles[lab])
        add_arrows(ax, np.real(G), np.imag(G), n_arrows=3)

    # critical point
    if style == "latex":
        ax.plot([-1],[0], marker="x", color="black", markersize=9, mew=2, label="(-1,0)")
    else:
        ax.scatter([-1],[0], s=70, marker="x", linewidths=2, label="kritischer Punkt (-1,0)")

    # marginal marker
    Gm = G0(w_star, Kcrit)
    if style == "latex":
        ax.plot([np.real(Gm)],[np.imag(Gm)], marker="o", color="black", markersize=5)
        ax.annotate(r"$\omega^*\approx %.3g\,\mathrm{rad/s}$" % w_star, (np.real(Gm), np.imag(Gm)),
                    textcoords="offset points", xytext=(10,10))
    else:
        ax.scatter([np.real(Gm)],[np.imag(Gm)], s=55)
        ax.annotate("ω*≈%.3g rad/s" % w_star, (np.real(Gm), np.imag(Gm)),
                    textcoords="offset points", xytext=(10,10))

    # min-distance circle
    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(-1 + min_dist*np.cos(theta), 0 + min_dist*np.sin(theta), linewidth=1)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, which="both", alpha=0.3 if style=="latex" else 0.35)
    ax.legend(loc="best", frameon=True)

    if zoom:
        ax.set_xlim(*xlim_zoom); ax.set_ylim(*ylim_zoom)
        ax.set_title("Abbildung 7 (neu) – Zoom um (-1,0)")
    else:
        ax.set_title("Abbildung 7 (neu): Nyquist-Ortskurven (ω>0)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Abbildung7_neu_%s.png" % out_prefix))
    fig.savefig(os.path.join(OUT_DIR, "Abbildung7_neu_%s.pdf" % out_prefix))
    plt.close(fig)
    if style == "latex":
        plt.rcdefaults()

# full and zoom
plot("modern", zoom=False, out_prefix="modern_full")
plot("latex",  zoom=False, out_prefix="latex_full")
plot("modern", zoom=True,  out_prefix="modern")
plot("latex",  zoom=True,  out_prefix="latex")
