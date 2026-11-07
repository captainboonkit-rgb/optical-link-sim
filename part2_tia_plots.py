# %% [markdown]
# # Optical Link Simulator — Part 2: TIA design in LTspice
#
# อ่านผลจำลอง LTspice ของ TIA (LTC6268-10, Rf = 10 kΩ, Cf = 0.08 pF, photodiode 0.5 pF)
# แล้ววาดกราฟสไตล์เดียวกับ Part 1
#
# ไฟล์ที่ใช้ (export จาก LTspice: File -> Export data as text):
#   ltspice/tia_step.txt  transient, .step param Cf list 0.03p 0.08p
#   ltspice/tia_ac.txt    AC analysis ที่ Cf = 0.08p (ถ้ายังไม่มีไฟล์ ส่วนนี้จะถูกข้าม)

# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": GRAY, "axes.labelcolor": "#0b0b0b",
    "xtick.color": GRAY, "ytick.color": GRAY,
})

def read_ltspice_txt(path):
    """อ่านไฟล์ text ที่ export จาก LTspice -> dict {ชื่อ step: list ของแถว [x, ค่า]}"""
    steps, label = {}, None
    with open(path, encoding="latin-1") as f:
        f.readline()                                    # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Step Information"):
                label = line.split(":", 1)[1].split("(")[0].strip()
                steps[label] = []
                continue
            if label is None:
                label = "run"
                steps[label] = []
            x, y = line.split("\t", 1)
            steps[label].append((float(x), y))
    return steps

# %% [markdown]
# ## 1. Step response
# ใส่กระแส 10 µA เป็นพัลส์ -> Vout ควรเป็น -10 µA x 10 kΩ = -100 mV (วงจรกลับเฟส)
# Cf = 0.08 pF นิ่งดี ส่วน Cf = 0.03 pF oscillate ต่อเนื่อง

# %%
step = read_ltspice_txt("ltspice/tia_step.txt")
tr = {k: (np.array([r[0] for r in v]), np.array([float(r[1]) for r in v]))
      for k, v in step.items()}

def step_metrics(t, v, t0=1e-9, t1=6e-9):
    """rise time 10-90%, overshoot และ settling (±2%) ของขาลงแรกในช่วง t0..t1"""
    m = (t >= t0) & (t <= t1)
    t, v = t[m], v[m]
    final = v[-1]
    tt = np.linspace(t[0], t[-1], 20000)
    vv = np.interp(tt, t, v)
    t10 = tt[np.argmax(vv <= 0.1 * final)]
    t90 = tt[np.argmax(vv <= 0.9 * final)]
    overshoot = (vv.min() - final) / final * 100
    outside = np.abs(vv - final) > 0.02 * abs(final)
    t_settle = tt[np.nonzero(outside)[0][-1]] - t0
    return (t90 - t10) * 1e9, overshoot, t_settle * 1e9, final * 1e3

t8, v8 = tr["Cf=80f"]
rise_ns, os_pct, settle_ns, final_mV = step_metrics(t8, v8)
print(f"Cf = 0.08 pF: final {final_mV:.1f} mV, rise 10-90% {rise_ns:.2f} ns, "
      f"overshoot {os_pct:.1f} %, settling (2%) {settle_ns:.2f} ns")

fig, ax = plt.subplots(figsize=(9, 4))
t3, v3 = tr["Cf=30f"]
ax.plot(t3 * 1e9, v3 * 1e3, color=ORANGE, lw=1.5, label="Cf = 0.03 pF (unstable)")
ax.plot(t8 * 1e9, v8 * 1e3, color=BLUE, lw=2, label="Cf = 0.08 pF (chosen)")
ax.axhline(-100, color=GRAY, lw=0.8, ls="--", label="Ideal: −10 µA × 10 kΩ = −100 mV")
ax.set_xlim(0, 30)
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Vout [mV]")
ax.set_title("TIA step response — 10 µA current pulse (LTspice, LTC6268-10)", pad=30)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig("figures/tia_step_response.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Frequency response (ถ้ามีไฟล์ ltspice/tia_ac.txt)
# Export จาก LTspice ได้ทั้งแบบ (dB, องศา) และแบบ (จริง, จินตภาพ) โค้ดรองรับทั้งสองแบบ

# %%
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_complex(cell):
    a, b = (float(x) for x in re.findall(NUM, cell)[:2])
    if "dB" in cell:
        return a                                       # already magnitude in dB
    return 20 * np.log10(np.hypot(a, b))

if os.path.exists("ltspice/tia_ac.txt"):
    ac = read_ltspice_txt("ltspice/tia_ac.txt")
    rows = next(iter(ac.values()))
    f = np.array([r[0] for r in rows])
    mag = np.array([parse_complex(r[1]) for r in rows])
    ref = mag[np.argmin(np.abs(f - 10e6))]               # passband level at 10 MHz
    peaking = mag.max() - ref
    bw = f[np.nonzero(mag >= ref - 3)[0][-1]]
    print(f"transimpedance {ref:.1f} dBΩ ({10 ** (ref / 20) / 1e3:.1f} kΩ), "
          f"-3 dB bandwidth {bw / 1e6:.0f} MHz, peaking {peaking:.2f} dB")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(f, mag, color=BLUE, lw=2)
    ax.axhline(ref - 3, color=GRAY, lw=0.8, ls="--")
    ax.axvline(bw, color=GRAY, lw=0.8, ls="--")
    ax.text(bw * 1.1, ref - 1.5, f"−3 dB @ {bw / 1e6:.0f} MHz", color="#0b0b0b", fontsize=9)
    ax.set_ylim(ref - 40, ref + 6)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transimpedance [dBΩ]")
    ax.set_title("TIA frequency response — Rf = 10 kΩ, Cf = 0.08 pF")
    fig.tight_layout()
    fig.savefig("figures/tia_frequency.png", dpi=150)
    plt.show()
else:
    print("ltspice/tia_ac.txt not found - skipping frequency response plot")
