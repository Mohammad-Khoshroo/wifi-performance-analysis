#!/usr/bin/env python3
"""Complete analysis and handover state machine for CA2 Phase 3."""
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from omnetpp.scave import results

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

with open(os.path.join(HERE, "..", "params.json")) as f:
    P = json.load(f)

D = P["D"]                  # 87 m
V = P["v"]                  # 1.25 m/s
BI = P["BI"] / 1000.0       # 0.1 s
N_CH = P["nCh"]             # 8
ALPHA = P["alpha"]          # 2.8
SIGMA = P["sigma"]          # 5 dB
T_WALK = D / V              # 69.6 s

N_DBM = -110.0              # RadioMedium backgroundNoise power
PT_MW = 20.0                # 20 mW
FC = 5.03e9                 # Hz (channel 1; channel 3 is 5.07e9)
C_LIGHT = 299792458.0
PL_D0 = 20.0 * math.log10(4.0 * math.pi * FC / C_LIGHT)
PT_DBM = 10.0 * math.log10(PT_MW)
SENSITIVITY_DBM = -85.0


def mean_power_dbm(d):
    return PT_DBM - PL_D0 - 10.0 * ALPHA * np.log10(d)


def load_vectors(config):
    df = results.read_result_files(os.path.join(RESULTS, f"{config}-*.vec"))
    vecs = results.get_vectors(df, include_itervars=True, include_runattrs=True)
    vecs["repetition"] = pd.to_numeric(vecs["repetition"])
    return vecs


def vector(vecs, rep, module, name):
    row = vecs[(vecs.repetition == rep) & (vecs.module == "Roaming." + module) & (vecs.name == name)]
    if row.empty:
        return np.array([]), np.array([])
    return row.iloc[0].vectime, row.iloc[0].vecvalue


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    m = float(np.mean(arr))
    if len(arr) < 2:
        return m, 0.0
    h = float(stats.sem(arr) * stats.t.ppf(0.975, df=len(arr) - 1))
    return m, h


def analyze_walk_like(vecs, label):
    print(f"\n=== {label} (5 reps) ===")
    outages, scan_starts, x_hos = [], [], []
    for rep in range(5):
        ta1, _ = vector(vecs, rep, "ap1.sink", "packets:vector(packetLength)")
        ta2, _ = vector(vecs, rep, "ap2.sink", "packets:vector(packetLength)")
        tr, vr = vector(vecs, rep, "host.wlan.agent", "sentRequest:vector")
        tac, vac = vector(vecs, rep, "host.wlan.agent", "acceptConfirm:vector")
        tch, vch = vector(vecs, rep, "host.wlan.radio", "radioChannel:vector")

        t_last1 = float(ta1[-1])
        t_first2 = float(ta2[0])
        outage = t_first2 - t_last1

        # Handover events occur after initial association (t > 5s)
        ho_scans = [float(t) for t in tr[vr == 1] if t > 5.0]
        ho_scan_start = ho_scans[0]
        x_ho = min(V * ho_scan_start, D)
        outages.append(outage)
        scan_starts.append(ho_scan_start)
        x_hos.append(x_ho)

        ho_reqs = [(round(float(t), 6), int(c)) for t, c in zip(tr, vr) if t > 5.0]
        ho_accs = [(round(float(t), 6), int(c)) for t, c in zip(tac, vac) if t > 5.0]
        print(f"rep={rep}: ap1_last={t_last1:.6f}s, ap2_first={t_first2:.6f}s, "
              f"outage={outage:.4f}s, first_ho_scan={ho_scan_start:.4f}s (x={x_ho:.2f}m)")
        if rep == 0 or len(ho_scans) > 1:
            print(f"   ho_reqs={ho_reqs}")
            print(f"   ho_accs={ho_accs}")
            ho_chs = [(round(float(t), 6), int(c)) for t, c in zip(tch, vch) if t >= ho_scan_start - 1e-6]
            print(f"   ho_channels={ho_chs}")
        else:
            ho_chs = [(float(t), int(c)) for t, c in zip(tch, vch) if t >= ho_scan_start - 1e-6]
            dwells = [(ho_chs[i][1], round((ho_chs[i + 1][0] - ho_chs[i][0]) * 1000, 1))
                      for i in range(len(ho_chs) - 1)]
            print(f"   channel_dwells_ms={dwells}")

    m_out, ci_out = ci95(outages)
    m_x, ci_x = ci95(x_hos)
    m_t, ci_t = ci95(scan_starts)
    print(f"{label} Outage: {m_out:.4f} +/- {ci_out:.4f} s")
    print(f"{label} x_ho:   {m_x:.2f} +/- {ci_x:.2f} m (t={m_t:.2f} +/- {ci_t:.2f} s)")
    return outages, scan_starts, x_hos


def simulate_state_machine(t1, p1, t2, p2, threshold_dbm, hysteresis_db, n_avg):
    """
    Handover state machine on merged Survey beacon streams.
    States: 'AP1', 'AP2'
    Rules:
    - Estimate each AP's power via simple moving average of last N received beacons.
    - Check handover only when current AP estimated power < T.
    - Switch only if other AP estimated power >= current AP estimated power + H.
    - Timeout: if no beacon from current AP for 3.5 * BI, connection is lost -> switch to other AP.
    """
    events = [(float(t), 1, float(p)) for t, p in zip(t1, p1)] + \
             [(float(t), 2, float(p)) for t, p in zip(t2, p2)]
    events.sort(key=lambda e: e[0])

    buf = {1: [], 2: []}
    last_seen = {1: None, 2: None}
    state = 1  # starts associated with ap1
    handovers = []  # list of (t, x, from_ap, to_ap, reason)
    timeout = 3.5 * BI

    for t, ap, p_dbm in events:
        # Check 3.5*BI timeout on current AP before processing incoming event
        if last_seen[state] is not None and (t - last_seen[state]) >= timeout:
            other = 2 if state == 1 else 1
            # Switch if other AP has been seen recently (within timeout)
            if last_seen[other] is not None and (t - last_seen[other]) < timeout:
                x = min(V * t, D)
                handovers.append((t, x, state, other, "timeout"))
                state = other

        buf[ap].append(p_dbm)
        if len(buf[ap]) > n_avg:
            buf[ap].pop(0)
        last_seen[ap] = t

        if len(buf[state]) == 0:
            continue
        est_curr = float(np.mean(buf[state]))
        other = 2 if state == 1 else 1
        if len(buf[other]) == 0:
            continue
        # Only consider other AP if not timed out itself
        if last_seen[other] is None or (t - last_seen[other]) >= timeout:
            continue
        est_other = float(np.mean(buf[other]))

        if est_curr < threshold_dbm and est_other >= est_curr + hysteresis_db:
            x = min(V * t, D)
            handovers.append((t, x, state, other, "hysteresis"))
            state = other

    return handovers


def draw_state_machine_diagram(out_path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # States
    ap1_box = mpatches.FancyBboxPatch((0.8, 1.6), 2.5, 1.3, boxstyle="round,pad=0.15",
                                      facecolor="#dceeff", edgecolor="#1f77b4", linewidth=2)
    ap2_box = mpatches.FancyBboxPatch((6.7, 1.6), 2.5, 1.3, boxstyle="round,pad=0.15",
                                      facecolor="#ffe8d6", edgecolor="#ff7f0e", linewidth=2)
    ax.add_patch(ap1_box)
    ax.add_patch(ap2_box)

    ax.text(2.05, 2.25, "Associated\nwith AP1", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(7.95, 2.25, "Associated\nwith AP2", ha="center", va="center", fontsize=12, fontweight="bold")

    # Start arrow
    ax.annotate("", xy=(0.75, 2.25), xytext=(0.15, 2.25),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(0.42, 2.5, "start", ha="center", va="bottom", fontsize=10)

    # Transition AP1 -> AP2 (top)
    ax.annotate("", xy=(6.6, 2.65), xytext=(3.4, 2.65),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2ca02c", connectionstyle="arc3,rad=-0.18"))
    ax.text(5.0, 3.55,
            r"$(\bar{P}_1 < T \;\wedge\; \bar{P}_2 \geq \bar{P}_1 + H)$  OR  (no beacon from AP1 for $3.5\,\mathrm{BI}$)",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.9))

    # Transition AP2 -> AP1 (bottom)
    ax.annotate("", xy=(3.4, 1.85), xytext=(6.6, 1.85),
                arrowprops=dict(arrowstyle="->", lw=2, color="#d62728", connectionstyle="arc3,rad=-0.18"))
    ax.text(5.0, 0.90,
            r"$(\bar{P}_2 < T \;\wedge\; \bar{P}_1 \geq \bar{P}_2 + H)$  OR  (no beacon from AP2 for $3.5\,\mathrm{BI}$)",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9))

    ax.set_title("Designed Handover State Machine (SMA of last N beacons, threshold T, hysteresis H)",
                 fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    walk = load_vectors("Walk")
    scan_n = load_vectors("ScanNeighbor")
    survey = load_vectors("Survey")

    _, _, walk_x_hos = analyze_walk_like(walk, "Walk")
    analyze_walk_like(scan_n, "ScanNeighbor")

    # Section 5.2: Q5-4 Survey power vs distance
    print("\n=== Section 5.2: Survey Power Analysis ===")
    d_grid = np.linspace(2.0, math.sqrt(D ** 2 + 4.0), 300)
    p_model = mean_power_dbm(d_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    survey_streams = []
    for rep in range(3):
        t1, snir1 = vector(survey, rep, "meter.wlan[0].radio", "minSnir:vector")
        t2, snir2 = vector(survey, rep, "meter.wlan[1].radio", "minSnir:vector")
        m1 = t1 <= T_WALK
        m2 = t2 <= T_WALK
        t1, snir1 = t1[m1], snir1[m1]
        t2, snir2 = t2[m2], snir2[m2]
        p1 = N_DBM + 10.0 * np.log10(snir1)
        p2 = N_DBM + 10.0 * np.log10(snir2)
        d1 = np.sqrt((V * t1) ** 2 + 4.0)
        d2 = np.sqrt((D - V * t2) ** 2 + 4.0)
        survey_streams.append((t1, p1, t2, p2))

        alpha_pt = 0.45 if rep == 0 else 0.25
        lbl1 = "Survey beacons (3 runs)" if rep == 0 else None
        axes[0].scatter(d1, p1, s=8, alpha=alpha_pt, color="#1f77b4", label=lbl1)
        axes[1].scatter(d2, p2, s=8, alpha=alpha_pt, color="#ff7f0e", label=lbl1)

    for idx, (ax, title) in enumerate(zip(axes, ["AP1 (Channel 1)", "AP2 (Channel 3)"])):
        ax.plot(d_grid, p_model, "k-", linewidth=2, label=rf"Model $\bar{{P}}(d)$, $\alpha={ALPHA}$")
        ax.axhline(SENSITIVITY_DBM, color="r", linestyle="--", linewidth=1.5, label="Sensitivity (-85 dBm)")
        ax.set_xscale("log")
        ax.set_xlabel("Distance d (m) [log scale]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Received Power (dBm)")
    fig.suptitle("Q5-4: Measured Beacon Power vs. Distance (Survey, t <= D/v)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q5_4_power_vs_distance.png"), dpi=200)
    plt.close()

    # Q5-5: Sticky client calculations
    x_equal = D / 2.0
    t_equal = x_equal / V
    m_x_ho, ci_x_ho = ci95(walk_x_hos)
    delay_m = m_x_ho - x_equal
    delay_s = delay_m / V
    print(f"Equal mean power point: x = {x_equal:.2f} m (t = {t_equal:.2f} s)")
    print(f"Walk mean scan start:   x = {m_x_ho:.2f} +/- {ci_x_ho:.2f} m")
    print(f"Sticky client delay:    {delay_m:.2f} m ({delay_s:.2f} s)")

    # Q5-6(b): Theoretical noiseless handover point for H = 6 dB
    h_test = 6.0
    ratio = 10.0 ** (h_test / (10.0 * ALPHA))
    x_h6 = (ratio / (1.0 + ratio)) * D
    p1_at_xh6 = mean_power_dbm(math.sqrt(x_h6 ** 2 + 4.0))
    p2_at_xh6 = mean_power_dbm(math.sqrt((D - x_h6) ** 2 + 4.0))
    p_mid = mean_power_dbm(math.sqrt(x_equal ** 2 + 4.0))
    print(f"\nMean power at midpoint x={x_equal:.2f}m: {p_mid:.2f} dBm")
    print(f"Q5-6(b) Noiseless H=6dB point: x* = {x_h6:.2f} m (P1={p1_at_xh6:.2f} dBm, P2={p2_at_xh6:.2f} dBm)")

    # Section 5.3: Evaluate state machine for T = -75 dBm
    T_THRESH = -75.0
    H_VALS = [0, 2, 4, 6, 8, 10]
    N_VALS = [1, 5, 10]
    print(f"\n=== Section 5.3: Handover State Machine (T = {T_THRESH} dBm) ===")
    draw_state_machine_diagram(os.path.join(PLOTS, "q5_6_state_machine.png"))

    plt.figure(figsize=(8, 4.8))
    for n_avg in N_VALS:
        mean_counts = []
        print(f"\n--- N = {n_avg} ---")
        for h in H_VALS:
            counts = []
            last_xs = []
            for rep, (t1, p1, t2, p2) in enumerate(survey_streams):
                hos = simulate_state_machine(t1, p1, t2, p2, T_THRESH, h, n_avg)
                counts.append(len(hos))
                last_xs.append(hos[-1][1] if hos else float("nan"))
            m_cnt = float(np.mean(counts))
            m_lx = float(np.mean(last_xs))
            mean_counts.append(m_cnt)
            print(f"H={h:2d} dB: counts={counts} (mean={m_cnt:.2f}), last_x={[round(x,2) for x in last_xs]} (mean={m_lx:.2f} m)")
            if n_avg == 10 and h == 6:
                for rep, (t1, p1, t2, p2) in enumerate(survey_streams):
                    hos = simulate_state_machine(t1, p1, t2, p2, T_THRESH, h, n_avg)
                    print(f"   detail rep={rep}: {hos}")
        plt.plot(H_VALS, mean_counts, "o-", linewidth=2, label=f"N = {n_avg}")

    plt.xlabel("Hysteresis H (dB)")
    plt.ylabel("Mean Number of Handovers")
    plt.title(f"Q5-7: Handovers vs. Hysteresis H (T = {T_THRESH} dBm)")
    plt.xticks(H_VALS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q5_7_handovers_vs_H.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
