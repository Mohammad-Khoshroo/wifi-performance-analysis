#!/usr/bin/env python3
"""Analysis and modeling script for CA2 Phase 2."""
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

L = P["L"]          # 650 B
N = P["n2"]         # 5
R_SLOW = P["Rslow"] # 9 Mbps
R_FAST = P["Rfast"] # 48 Mbps
RATES = [R_SLOW] + [R_FAST] * (N - 1)
T_MEAS = 10.0       # s (12s - 2s warmup)

# 802.11a constants
CW_MIN = 15
SLOT = 9.0          # us
SIFS = 16.0         # us
DIFS = SIFS + 2 * SLOT  # 34 us
AVG_BACKOFF = (CW_MIN / 2.0) * SLOT  # 67.5 us

# Headers added to L=650 B UDP payload:
# UDP (8 B) + IPv4 (20 B) + LLC/EPD (2 B) + MAC hdr (24 B) + FCS (4 B) = 58 B -> B_DATA = 708 B
# Let's also verify from simulation scalars!
B_UDP = 8
B_IP = 20
B_LLC = 2   # Ieee802EpdHeader (2 B) when using Ipv4 + Udp in INET 4.7
B_MAC_HDR = 24
B_FCS = 4
B_DATA = L + B_UDP + B_IP + B_LLC + B_MAC_HDR + B_FCS  # 708 B
B_ACK = 14  # 10 B + 4 B FCS

MANDATORY_RATES = [6, 12, 24]


def r_ack_for(r):
    return max(m for m in MANDATORY_RATES if m <= r)


def t_ppdu(B, r):
    """802.11a OFDM PPDU duration in microseconds."""
    n_dbps = 4 * r
    return 20.0 + 4.0 * math.ceil((16 + 8 * B + 6) / n_dbps)


T_DATA = [t_ppdu(B_DATA, r) for r in RATES]
T_ACK = [t_ppdu(B_ACK, r_ack_for(r)) for r in RATES]


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    se = float(np.std(arr, ddof=1)) / math.sqrt(n)
    half = float(stats.t.ppf(0.975, df=n - 1) * se)
    return mean, half


def jain(vals):
    vals = np.asarray(vals, dtype=float)
    s1 = np.sum(vals)
    s2 = np.sum(vals ** 2)
    return float((s1 ** 2) / (len(vals) * s2)) if s2 > 0 else 0.0


def load_config(config):
    df = results.read_result_files(os.path.join(RESULTS, f"{config}-*.sca"))
    sc = results.get_scalars(df, include_itervars=True, include_runattrs=True)
    st = results.get_histograms(df, include_itervars=True, include_runattrs=True)
    for t in (sc, st):
        if "repetition" in t:
            t["repetition"] = pd.to_numeric(t["repetition"])
    return sc, st


def analyze_config(config, direction):
    sc, st = load_config(config)
    counts = {}
    for i in range(N):
        mod = f"Anomaly.server.app[{i}]" if direction == "up" else f"Anomaly.host[{i}].app[0]"
        rows = sc[(sc["module"] == mod) & (sc["name"] == "packetReceived:count")]
        counts[i] = rows.set_index("repetition")["value"].sort_index()
    df_counts = pd.DataFrame(counts)
    df_tp = df_counts * (8.0 * L / T_MEAS / 1e6)

    # Airtime share a_i = N_i * (T_DATA_i + SIFS + T_ACK_i) / T_MEAS_us
    t_meas_us = T_MEAS * 1e6
    df_air = pd.DataFrame({
        i: df_counts[i] * (T_DATA[i] + SIFS + T_ACK[i]) / t_meas_us
        for i in range(N)
    })

    # End-to-end delay histograms (mean in seconds -> ms)
    d = st[st["name"] == "endToEndDelay:histogram"]
    delay_ms = {}
    for i in range(N):
        mod = f"Anomaly.server.app[{i}]" if direction == "up" else f"Anomaly.host[{i}].app[0]"
        rows = d[d["module"] == mod]
        delay_ms[i] = ci95(rows["mean"].values * 1e3) if len(rows) > 0 else (np.nan, np.nan)

    tp_stats = {i: ci95(df_tp[i].values) for i in range(N)}
    tot_tp = ci95(df_tp.sum(axis=1).values)
    air_stats = {i: ci95(df_air[i].values) for i in range(N)}

    j_tp_runs = [jain(df_tp.loc[r].values) for r in df_tp.index]
    j_air_runs = [jain(df_air.loc[r].values) for r in df_air.index]

    return {
        "df_counts": df_counts,
        "df_tp": df_tp,
        "df_air": df_air,
        "tp_stats": tp_stats,
        "tot_tp": tot_tp,
        "air_stats": air_stats,
        "jain_tp": ci95(j_tp_runs),
        "jain_air": ci95(j_air_runs),
        "delay_ms": delay_ms,
        "sc": sc,
    }


def main():
    # Verify actual MPDU bytes at MAC in DownlinkFifo
    sc_fifo, _ = load_config("DownlinkFifo")
    mac_rows = sc_fifo[sc_fifo["module"] == "Anomaly.ap.wlan[0].mac.dcf"]
    sent_cnt = mac_rows[mac_rows["name"] == "packetSentToPeer:count"]["value"].mean()
    # Also check app packetSent length if available
    print("=== Q4-1: Frame Sizes & Durations ===")
    print(f"L={L} B, UDP=8 B, IPv4=20 B, LLC/EPD=2 B, MAC+FCS=28 B => B_DATA={B_DATA} B, B_ACK={B_ACK} B")
    print(f"Slow (host[0]): R={R_SLOW} Mbps, R_ACK={r_ack_for(R_SLOW)} Mbps => T_DATA={T_DATA[0]:.1f} us, T_ACK={T_ACK[0]:.1f} us")
    print(f"Fast (host[1..4]): R={R_FAST} Mbps, R_ACK={r_ack_for(R_FAST)} Mbps => T_DATA={T_DATA[1]:.1f} us, T_ACK={T_ACK[1]:.1f} us")
    print(f"(Check: DownlinkFifo AP packetSentToPeer:count mean = {sent_cnt:.1f})")

    # Q4-2: Uplink & DownlinkFifo
    res_up = analyze_config("Uplink", "up")
    res_fifo = analyze_config("DownlinkFifo", "down")

    print("\n=== Q4-2: Uplink & DownlinkFifo ===")
    for label, res in [("Uplink", res_up), ("DownlinkFifo", res_fifo)]:
        print(f"\n[{label}]")
        for i in range(N):
            m, h = res["tp_stats"][i]
            am, ah = res["air_stats"][i]
            print(f"  host[{i}] ({RATES[i]} Mbps): S = {m:.4f} ± {h:.4f} Mbps | a = {am*100:.2f}% ± {ah*100:.2f}%")
        tm, th = res["tot_tp"]
        jtm, jth = res["jain_tp"]
        jam, jah = res["jain_air"]
        print(f"  Total S = {tm:.4f} ± {th:.4f} Mbps")
        print(f"  Jain(S) = {jtm:.4f} ± {jth:.4f} | Jain(a) = {jam:.4f} ± {jah:.4f}")
        print(f"  Total airtime sum = {sum(res['air_stats'][i][0] for i in range(N))*100:.2f}%")

    # Q4-3: Analytical model for DownlinkFifo
    t_exch = [DIFS + AVG_BACKOFF + T_DATA[i] + SIFS + T_ACK[i] for i in range(N)]
    s_fifo_model = (8.0 * L) / sum(t_exch)
    s_fast_only_model = (8.0 * L) / sum(t_exch[1:])
    pct_drop_fast = (1.0 - s_fifo_model / s_fast_only_model) * 100.0
    print("\n=== Q4-3: DownlinkFifo Analytical Model ===")
    print(f"T_exch_slow = {t_exch[0]:.1f} us, T_exch_fast = {t_exch[1]:.1f} us")
    print(f"Model S_i (with slow) = {s_fifo_model:.4f} Mbps (Total = {N * s_fifo_model:.4f} Mbps)")
    print(f"Sim   S_fast_avg      = {np.mean([res_fifo['tp_stats'][i][0] for i in range(1, N)]):.4f} Mbps, S_slow = {res_fifo['tp_stats'][0][0]:.4f} Mbps")
    print(f"Model S_i (without slow, 4 fast) = {s_fast_only_model:.4f} Mbps (Total = {4 * s_fast_only_model:.4f} Mbps)")
    print(f"Fast station throughput reduction due to slow station = {pct_drop_fast:.2f}%")

    # Q4-4: Analytical model for DownlinkAirtime
    denom_air = sum(t_exch[j] / T_DATA[j] for j in range(N))
    s_air_model = [(8.0 * L / T_DATA[i]) / denom_air for i in range(N)]
    print("\n=== Q4-4: DownlinkAirtime Analytical Model ===")
    print(f"Slow S_0 = {s_air_model[0]:.4f} Mbps, Fast S_1..4 = {s_air_model[1]:.4f} Mbps, Total = {sum(s_air_model):.4f} Mbps")

    # Q4-5: DownlinkAirtime simulation
    res_air = analyze_config("DownlinkAirtime", "down")
    print("\n=== Q4-5: DownlinkAirtime Simulation vs Model ===")
    for i in range(N):
        m, h = res_air["tp_stats"][i]
        am, ah = res_air["air_stats"][i]
        rel_err = abs(m - s_air_model[i]) / s_air_model[i] * 100.0
        print(f"  host[{i}] ({RATES[i]} Mbps): Sim={m:.4f}±{h:.4f} Mbps | Model={s_air_model[i]:.4f} Mbps | Err={rel_err:.2f}% | a={am*100:.2f}%")
    tm_air, th_air = res_air["tot_tp"]
    tm_fifo, _ = res_fifo["tot_tp"]
    inc_total = (tm_air - tm_fifo) / tm_fifo * 100.0
    dec_slow = (res_fifo["tp_stats"][0][0] - res_air["tp_stats"][0][0]) / res_fifo["tp_stats"][0][0] * 100.0
    print(f"  Total S = {tm_air:.4f} ± {th_air:.4f} Mbps (vs FIFO: +{inc_total:.2f}%)")
    print(f"  Slow station throughput decrease = {dec_slow:.2f}%")
    print(f"  Jain(S) = {res_air['jain_tp'][0]:.4f} ± {res_air['jain_tp'][1]:.4f} | Jain(a) = {res_air['jain_air'][0]:.4f} ± {res_air['jain_air'][1]:.4f}")

    # Q4-6: 10-step manual trace of schedulePacket() for 1 slow + 1 fast, quantum=500 us
    print("\n=== Q4-6: First 10 decisions of schedulePacket() (quantum=500 us) ===")
    q_us = 500.0
    t_s, t_f = T_DATA[0], T_DATA[1]
    deficit = [0.0, 0.0]
    cur = 0
    step = 0
    while step < 10:
        if deficit[cur] <= 0.0:
            deficit[cur] += q_us
            cur = (cur + 1) % 2
        else:
            step += 1
            before = list(deficit)
            cost = t_s if cur == 0 else t_f
            deficit[cur] -= cost
            print(f"  Decision {step:2d}: serve host[{cur}] ({'slow' if cur==0 else 'fast'}, {cost:.0f} us) | deficit before={before} -> after={[round(x,1) for x in deficit]}")

    # Q4-7: Little's Law vs simulated endToEndDelay
    print("\n=== Q4-7: End-to-End Delay (Little's Law vs Simulation) ===")
    for label, res, q_cap_slow, q_cap_fast in [
        ("DownlinkFifo (Q=100 shared)", res_fifo, 100.0 / N, 100.0 / N),
        ("DownlinkAirtime (Q=50 per-sta)", res_air, 50.0, 50.0),
    ]:
        print(f"  [{label}]")
        for i, q_len in [(0, q_cap_slow), (1, q_cap_fast)]:
            lam = res["tp_stats"][i][0] * 1e6 / (8.0 * L)  # pkt/s
            w_little_ms = (q_len / lam) * 1e3
            dm, dh = res["delay_ms"][i]
            print(f"    host[{i}]: Little={w_little_ms:.2f} ms | Sim={dm:.2f} ± {dh:.2f} ms")

    # Q4-8: Low load slow station
    res_fifo_ll = analyze_config("DownlinkFifoLowLoad", "down")
    res_air_ll = analyze_config("DownlinkAirtimeLowLoad", "down")
    req_slow_mbps = (8.0 * L / 0.025) / 1e6
    print(f"\n=== Q4-8: Low-load slow station (requested = {req_slow_mbps:.4f} Mbps) ===")
    for label, res in [("DownlinkFifoLowLoad", res_fifo_ll), ("DownlinkAirtimeLowLoad", res_air_ll)]:
        print(f"  [{label}]")
        for i in range(N):
            m, h = res["tp_stats"][i]
            dm, dh = res["delay_ms"][i]
            print(f"    host[{i}]: S={m:.4f}±{h:.4f} Mbps | delay={dm:.2f}±{dh:.2f} ms")
        print(f"    Total S = {res['tot_tp'][0]:.4f} ± {res['tot_tp'][1]:.4f} Mbps")

    # Plots
    x = np.arange(N)
    labels = [f"host[{i}]\n({RATES[i]}M)" for i in range(N)]
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(x - w, [res_up["tp_stats"][i][0] for i in range(N)], w,
            yerr=[res_up["tp_stats"][i][1] for i in range(N)], capsize=3, label="Uplink")
    ax1.bar(x, [res_fifo["tp_stats"][i][0] for i in range(N)], w,
            yerr=[res_fifo["tp_stats"][i][1] for i in range(N)], capsize=3, label="Downlink FIFO")
    ax1.bar(x + w, [res_air["tp_stats"][i][0] for i in range(N)], w,
            yerr=[res_air["tp_stats"][i][1] for i in range(N)], capsize=3, label="Downlink Airtime DRR")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title("Per-Station Throughput")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - w, [res_up["air_stats"][i][0] * 100 for i in range(N)], w, label="Uplink")
    ax2.bar(x, [res_fifo["air_stats"][i][0] * 100 for i in range(N)], w, label="Downlink FIFO")
    ax2.bar(x + w, [res_air["air_stats"][i][0] * 100 for i in range(N)], w, label="Downlink Airtime DRR")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Channel Airtime Share (%)")
    ax2.set_title("Per-Station Airtime Share ($a_i$)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q4_throughput_and_airtime.png"), dpi=160)
    plt.close()

    # Plot Q4-8 comparison
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.18, [res_fifo_ll["tp_stats"][i][0] for i in range(N)], 0.36,
           yerr=[res_fifo_ll["tp_stats"][i][1] for i in range(N)], capsize=3, label="Downlink FIFO (Low Load)")
    ax.bar(x + 0.18, [res_air_ll["tp_stats"][i][0] for i in range(N)], 0.36,
           yerr=[res_air_ll["tp_stats"][i][1] for i in range(N)], capsize=3, label="Downlink Airtime DRR (Low Load)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Low-Load Slow Station (host[0] = exp(25ms))")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "q4_8_low_load.png"), dpi=160)
    plt.close()
    print("\nSaved plots to phase2/plots/")


if __name__ == "__main__":
    main()
