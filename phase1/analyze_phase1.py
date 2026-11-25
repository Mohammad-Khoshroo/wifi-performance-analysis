#!/usr/bin/env python3
"""Analysis and modeling script for CA2 Phase 1."""
import json
import math
import os

import matplotlib.pyplot as plt
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

L = P["L"]          # payload in bytes (650)
R = P["R"]          # data bitrate in Mbps (12)
T_MEAS = 10.0       # measurement interval in seconds (12s - 2s warmup)

# 802.11a constants
CW_MIN = 15
CW_MAX = 1023
SLOT = 9.0          # us
SIFS = 16.0         # us
DIFS = SIFS + 2 * SLOT  # 34 us

B_DATA = L + 3 + 28     # 650 + 3 (LLC) + 28 (MAC hdr + FCS) = 681 B
B_ACK = 14              # 10 (ACK hdr) + 4 (FCS) = 14 B
B_RTS = 20              # B
B_CTS = 14              # B

# 802.11a mandatory rates: 6, 12, 24 Mbps
MANDATORY_RATES = [6, 12, 24]
R_ACK = max(r for r in MANDATORY_RATES if r <= R)


def t_ppdu(B, r):
    """802.11a OFDM PPDU duration in microseconds."""
    n_dbps = 4 * r
    return 20.0 + 4.0 * math.ceil((16 + 8 * B + 6) / n_dbps)


T_DATA = t_ppdu(B_DATA, R)
T_ACK = t_ppdu(B_ACK, R_ACK)
T_RTS = t_ppdu(B_RTS, R_ACK)
T_CTS = t_ppdu(B_CTS, R_ACK)


def slotted_throughput(n, tau):
    """Slotted contention model throughput in Mbps."""
    p_idle = (1.0 - tau) ** n
    p_s = n * tau * ((1.0 - tau) ** (n - 1))
    p_c = 1.0 - p_idle - p_s
    t_s = DIFS + T_DATA + SIFS + T_ACK
    t_c = DIFS + T_DATA
    # L in bytes, durations in us -> bits / us = Mbps
    return (p_s * 8.0 * L) / (p_idle * SLOT + p_s * t_s + p_c * t_c)


def load_scalars(config):
    df = results.read_result_files(os.path.join(RESULTS, f"{config}-*.sca"))
    sc = results.get_scalars(df, include_itervars=True, include_runattrs=True)
    for col in ("n", "repetition"):
        if col in sc:
            sc[col] = pd.to_numeric(sc[col])
    return sc


def scalar_sum(sc, module_regex, name):
    rows = sc[(sc["name"] == name) & sc["module"].str.fullmatch(module_regex)]
    keys = [c for c in ("n", "rtsThreshold", "repetition") if c in rows and rows[c].notna().all()]
    return rows.groupby(keys)["value"].sum()


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr)
    if len(arr) < 2:
        return mean, 0.0
    se = stats.sem(arr)
    h = se * stats.t.ppf(0.975, df=len(arr) - 1)
    return mean, h


def main():
    print(f"=== Group Parameters: students={P['students']}, L={L} B, R={R} Mbps ===")
    print(f"B_DATA={B_DATA} B, B_ACK={B_ACK} B, R_ACK={R_ACK} Mbps")
    print(f"T_DATA={T_DATA:.1f} us, T_ACK={T_ACK:.1f} us, T_RTS={T_RTS:.1f} us, T_CTS={T_CTS:.1f} us")
    k_bar = CW_MIN / 2.0
    s1 = (8.0 * L) / (DIFS + k_bar * SLOT + T_DATA + SIFS + T_ACK)
    print(f"S1 (analytical, n=1, k_bar={k_bar}) = {s1:.6f} Mbps")

    # Section 3.2: Slotted model with tau0
    tau0 = 2.0 / (CW_MIN + 2.0)
    ns_model = np.arange(1, 51)
    s_tau0_50 = [slotted_throughput(n, tau0) for n in ns_model]
    best_idx = int(np.argmax(s_tau0_50))
    print(f"\n=== Section 3.2: Slotted Model (tau0 = {tau0:.6f}) ===")
    print(f"S(1, tau0) = {s_tau0_50[0]:.6f} Mbps (matches S1={s1:.6f})")
    print(f"Max throughput = {s_tau0_50[best_idx]:.6f} Mbps at n = {ns_model[best_idx]}")
    print(f"tau0 > 1/n for n > {1.0/tau0:.2f} -> n >= {math.ceil(1.0/tau0)}")

    # Section 3.3: Sweep results
    sweep = load_scalars("Sweep")
    n_rx = scalar_sum(sweep, r"Contention\.ap\.sink", "packets:count")
    n_first = scalar_sum(sweep, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetSentToPeerWithoutRetry:count")
    n_retry = scalar_sum(sweep, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetSentToPeerWithRetry:count")
    n_drop = scalar_sum(sweep, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetDropRetryLimitReached:count")

    s_sim_df = (8.0 * L * n_rx) / (T_MEAS * 1e6)  # Mbps
    p_hat_df = (n_retry + n_drop) / (n_retry + n_first)

    ns_sweep = sorted(s_sim_df.index.get_level_values("n").unique())
    sim_means, sim_cis = [], []
    p_hat_means, tau_hat_means = [], []
    s_tau0_sweep, s_tauhat_sweep = [], []

    print("\n=== Section 3.3: Sweep Comparison ===")
    print(f"{'n':>3} | {'S_sim (Mbps)':>18} | {'S(tau0)':>9} | {'S(tau_hat)':>10} | {'p_hat':>8} | {'tau_hat':>8} | {'1/n':>8}")
    for n in ns_sweep:
        vals = s_sim_df.xs(n, level="n").values
        m, h = ci95(vals)
        sim_means.append(m)
        sim_cis.append(h)

        p_vals = p_hat_df.xs(n, level="n").values
        p_m = float(np.mean(p_vals))
        p_hat_means.append(p_m)
        if n > 1:
            tau_h = 1.0 - (1.0 - p_m) ** (1.0 / (n - 1))
        else:
            tau_h = tau0
        tau_hat_means.append(tau_h)
        s0 = slotted_throughput(n, tau0)
        sh = slotted_throughput(n, tau_h)
        s_tau0_sweep.append(s0)
        s_tauhat_sweep.append(sh)
        print(f"{n:3d} | {m:7.4f} +/- {h:6.4f} | {s0:9.4f} | {sh:10.4f} | {p_m:8.4f} | {tau_h:8.4f} | {1.0/n:8.4f}")

    # Relative errors at n = 30
    idx30 = ns_sweep.index(30)
    rel_err_tau0 = abs(s_tau0_sweep[idx30] - sim_means[idx30]) / sim_means[idx30] * 100
    rel_err_tauhat = abs(s_tauhat_sweep[idx30] - sim_means[idx30]) / sim_means[idx30] * 100
    print(f"\nRelative error at n=30: tau0={rel_err_tau0:.2f}%, tau_hat={rel_err_tauhat:.2f}%")

    # Plot 1: Slotted model n=1..50 (Q3-4)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ns_model, s_tau0_50, "b-", linewidth=2, label=r"Slotted model ($\tau_0 = 2/17$)")
    plt.axvline(ns_model[best_idx], color="gray", linestyle="--", alpha=0.7,
                label=f"Peak at n={ns_model[best_idx]} ({s_tau0_50[best_idx]:.3f} Mbps)")
    plt.xlabel("Number of stations (n)")
    plt.ylabel("Throughput S (Mbps)")
    plt.title(f"Slotted Contention Model (L={L} B, R={R} Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q3_4_slotted_model.png"), dpi=200)
    plt.close()

    # Plot 2: Simulation vs Models (Q3-7)
    plt.figure(figsize=(8, 5))
    plt.errorbar(ns_sweep, sim_means, yerr=sim_cis, fmt="o-", capsize=4, linewidth=2,
                 label="Simulation (95% CI)")
    plt.plot(ns_sweep, s_tau0_sweep, "s--", linewidth=1.8, label=r"Model ($\tau = \tau_0$)")
    plt.plot(ns_sweep, s_tauhat_sweep, "^-.", linewidth=1.8, label=r"Model ($\tau = \hat{\tau}$)")
    plt.xlabel("Number of stations (n)")
    plt.ylabel("Throughput S (Mbps)")
    plt.title(f"Simulation vs. Slotted Contention Model (L={L} B, R={R} Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q3_7_model_vs_sim.png"), dpi=200)
    plt.close()

    # Plot 3: tau_hat vs tau0 vs 1/n (Q3-7)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ns_sweep[1:], tau_hat_means[1:], "o-", linewidth=2, label=r"$\hat{\tau}$ (from simulation)")
    plt.axhline(tau0, color="r", linestyle="--", linewidth=1.5, label=rf"$\tau_0 = {tau0:.4f}$")
    plt.plot(ns_sweep[1:], [1.0 / n for n in ns_sweep[1:]], "g-.", linewidth=1.8, label=r"Optimal $1/n$")
    plt.xlabel("Number of stations (n)")
    plt.ylabel(r"Transmission probability $\tau$")
    plt.title(r"Estimated $\hat{\tau}$ vs. $\tau_0$ and $1/n$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q3_7_tau_comparison.png"), dpi=200)
    plt.close()

    # Section 3.4: Hidden terminal & RTS/CTS
    print("\n=== Section 3.4: Hidden Terminal & RTS/CTS ===")
    f_c = 5.17e9
    c_light = 299792458.0
    lam = c_light / f_c
    pt_mw = 20.0
    for dist_name, d in [("Station -> AP (100m)", 100.0), ("Station <-> Station (200m)", 200.0)]:
        pr_mw = pt_mw * (lam / (4 * math.pi * d)) ** 2
        pr_dbm = 10.0 * math.log10(pr_mw)
        print(f"{dist_name}: P_r = {pr_dbm:.2f} dBm (threshold = -77.00 dBm)")

    hidden = load_scalars("Hidden")
    h_ap_rx = scalar_sum(hidden, r"Contention\.ap\.sink", "packets:count")
    h_sta0 = scalar_sum(hidden, r"Contention\.host\[0\]\.wlan\.mac\.dcf", "packetReceivedFromPeer:count")
    h_sta1 = scalar_sum(hidden, r"Contention\.host\[1\]\.wlan\.mac\.dcf", "packetReceivedFromPeer:count")
    h_first = scalar_sum(hidden, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetSentToPeerWithoutRetry:count")
    h_retry = scalar_sum(hidden, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetSentToPeerWithRetry:count")
    h_drop = scalar_sum(hidden, r"Contention\.host\[\d+\]\.wlan\.mac\.dcf", "packetDropRetryLimitReached:count")

    for rts in ["3000B", "100B"]:
        s_tot = (8.0 * L * h_ap_rx.xs(rts, level="rtsThreshold").values) / (T_MEAS * 1e6)
        div = 2.0 if rts == "100B" else 1.0
        s0_vals = (8.0 * L * (h_sta0.xs(rts, level="rtsThreshold").values / div)) / (T_MEAS * 1e6)
        s1_vals = (8.0 * L * (h_sta1.xs(rts, level="rtsThreshold").values / div)) / (T_MEAS * 1e6)
        j_vals = ((s0_vals + s1_vals) ** 2) / (2.0 * (s0_vals ** 2 + s1_vals ** 2))
        p_vals = (h_retry.xs(rts, level="rtsThreshold").values + h_drop.xs(rts, level="rtsThreshold").values) / (
            h_retry.xs(rts, level="rtsThreshold").values + h_first.xs(rts, level="rtsThreshold").values
        )
        m_tot, ci_tot = ci95(s_tot)
        m_0, ci_0 = ci95(s0_vals)
        m_1, ci_1 = ci95(s1_vals)
        m_j, ci_j = ci95(j_vals)
        m_p, ci_p = ci95(p_vals)
        print(f"rtsThreshold={rts}: S_tot={m_tot:.4f}+/-{ci_tot:.4f} Mbps, "
              f"S0={m_0:.4f}+/-{ci_0:.4f}, S1={m_1:.4f}+/-{ci_1:.4f}, "
              f"Jain={m_j:.4f}+/-{ci_j:.4f}, p_hat={m_p:.4f}")

    # Q3-10 Overhead vs collision savings
    overhead_us = T_RTS + T_CTS + 2 * SIFS
    savings_us = T_DATA - T_RTS
    p_no_rts = float(np.mean(
        (h_retry.xs("3000B", level="rtsThreshold").values + h_drop.xs("3000B", level="rtsThreshold").values) /
        (h_retry.xs("3000B", level="rtsThreshold").values + h_first.xs("3000B", level="rtsThreshold").values)
    ))
    # Each collision involves 2 failed frames:
    # For N_succ successful frames and N_coll collisions (2*N_coll failed frames):
    # p_hat = 2*N_coll / (N_succ + 2*N_coll) => N_coll / N_succ = p_hat / (2 * (1 - p_hat))
    coll_per_succ = p_no_rts / (2.0 * (1.0 - p_no_rts))
    expected_saving_per_succ = coll_per_succ * savings_us
    print(f"\nQ3-10: RTS/CTS overhead per success = {overhead_us:.1f} us, "
          f"savings per collision = {savings_us:.1f} us")
    print(f"p_hat (no RTS) = {p_no_rts:.4f}, collisions per success = {coll_per_succ:.4f}, "
          f"expected saving per success = {expected_saving_per_succ:.1f} us")


if __name__ == "__main__":
    main()
