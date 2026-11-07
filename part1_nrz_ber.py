# %% [markdown]
# # Optical Link Simulator — Part 1: NRZ eye diagram & BER
#
# จำลองลิงก์ไฟเบอร์แบบ NRZ (on-off keying) ความเร็ว 10 Gb/s
#   laser (P1/P0, extinction ratio) -> fiber -> PIN photodiode -> TIA (bandwidth + noise) -> decision
#
# ผลลัพธ์ (บันทึกลงโฟลเดอร์ figures/):
#   1. eye diagram ที่กำลังแสงสูงและต่ำ
#   2. BER vs received power: Monte Carlo เทียบกับสูตรทฤษฎี (Q-factor)
#   3. receiver sensitivity ที่ BER = 1e-12
#   4. link budget: ไฟเบอร์ยาวได้สูงสุดเท่าไหร่
#
# รันใน VS Code: กด "Run Cell" ทีละ cell หรือรันทั้งไฟล์ด้วย  python part1_nrz_ber.py

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import signal, special, optimize

os.makedirs("figures", exist_ok=True)
q = 1.602176634e-19           # electron charge [C]

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": GRAY, "axes.labelcolor": "#0b0b0b",
    "xtick.color": GRAY, "ytick.color": GRAY,
})

# %% [markdown]
# ## 1. พารามิเตอร์ของลิงก์
# ค่าเหล่านี้เป็นค่าสมมติที่ใกล้เคียงตัวรับ PIN + TIA สำหรับ 10G datacom ที่ 1310 nm
# ลองเปลี่ยนค่าแล้วรันใหม่เพื่อดูว่าผลเปลี่ยนยังไง

# %%
Rb = 10e9              # bit rate [b/s]
sps = 16               # samples per bit (oversampling)
R = 0.85               # photodiode responsivity [A/W]
ER_dB = 6.0            # extinction ratio P1/P0 [dB]
B = 0.75 * Rb          # receiver bandwidth [Hz] (ใช้เป็นทั้ง 3-dB cutoff และ noise bandwidth)
in_density = 15e-12    # TIA input-referred noise current density [A/sqrt(Hz)]
BER_target = 1e-12

fs = Rb * sps          # simulation sample rate [Hz]

# %% [markdown]
# ## 2. ทฤษฎี: ระดับสัญญาณ, noise และ BER
# - กำลังแสงเฉลี่ย Pavg กับ extinction ratio r = P1/P0  ->  P1 = 2*Pavg*r/(r+1),  P0 = 2*Pavg/(r+1)
# - กระแส photodiode: I = R * P
# - noise ที่ตัวรับ:  sigma^2 = sigma_thermal^2 + sigma_shot^2 = in^2 * B + 2*q*I*B
#   (shot noise ขึ้นกับกระแส ดังนั้นบิต 1 มี noise มากกว่าบิต 0)
# - Q = (I1 - I0) / (sigma1 + sigma0),   BER = 0.5 * erfc(Q / sqrt(2))
# - เกณฑ์ตัดสินที่ดีที่สุด: I_D = (sigma0*I1 + sigma1*I0) / (sigma0 + sigma1)

# %%
def dbm_to_w(p_dbm):
    return 1e-3 * 10 ** (np.asarray(p_dbm) / 10)

def power_levels(P_avg):
    r = 10 ** (ER_dB / 10)
    return 2 * P_avg * r / (r + 1), 2 * P_avg / (r + 1)

def noise_sigma(I):
    return np.sqrt(in_density ** 2 * B + 2 * q * np.asarray(I) * B)

def current_levels(P_avg_dBm):
    P1, P0 = power_levels(dbm_to_w(P_avg_dBm))
    return R * P1, R * P0

def q_factor(P_avg_dBm):
    I1, I0 = current_levels(P_avg_dBm)
    return (I1 - I0) / (noise_sigma(I1) + noise_sigma(I0))

def ber_theory(P_avg_dBm):
    return 0.5 * special.erfc(q_factor(P_avg_dBm) / np.sqrt(2))

def decision_threshold(P_avg_dBm):
    I1, I0 = current_levels(P_avg_dBm)
    s1, s0 = noise_sigma(I1), noise_sigma(I0)
    return (s0 * I1 + s1 * I0) / (s0 + s1)

Q_target = np.sqrt(2) * special.erfcinv(2 * BER_target)
sensitivity_dBm = optimize.brentq(lambda p: q_factor(p) - Q_target, -40, 0)
print(f"thermal noise (rms)      : {noise_sigma(0) * 1e6:.2f} uA")
print(f"Q needed for BER {BER_target:.0e} : {Q_target:.2f}")
print(f"receiver sensitivity     : {sensitivity_dBm:.2f} dBm  (theory, no ISI)")

# %% [markdown]
# ## 3. จำลองสัญญาณ (Monte Carlo)
# 1. สุ่มบิต 0/1 แล้วแปลงเป็นกำลังแสง NRZ (P1/P0) ความละเอียด sps จุดต่อบิต
# 2. photodiode แปลงเป็นกระแส แล้วผ่าน Bessel low-pass ลำดับ 4 แทน bandwidth ของตัวรับ
#    (ทำให้ขอบสัญญาณช้าลงและเกิด ISI เล็กน้อย)
# 3. บวก noise แบบ Gaussian ที่ผ่าน filter เดียวกัน ขนาดตาม sigma(I) ในแต่ละจุด
# 4. สุ่มตัวอย่างกลางบิต เทียบกับ I_D แล้วนับบิตที่ผิด

# %%
b_rx, a_rx = signal.bessel(4, B, btype="low", fs=fs, norm="mag")
rng = np.random.default_rng(seed=1)

def simulate_link(P_avg_dBm, n_bits):
    bits = rng.integers(0, 2, n_bits)
    P1, P0 = power_levels(dbm_to_w(P_avg_dBm))
    i_pd = R * np.repeat(np.where(bits == 1, P1, P0), sps)        # ideal photocurrent
    i_clean = signal.lfilter(b_rx, a_rx, i_pd)                     # receiver bandwidth
    n = signal.lfilter(b_rx, a_rx, rng.standard_normal(i_pd.size))
    n /= n.std()                                                   # unit-variance band-limited noise
    i_rx = i_clean + noise_sigma(np.clip(i_clean, 0, None)) * n
    return bits, i_clean, i_rx

def find_sampling_offset():
    """หาจุดสุ่มตัวอย่างที่ตาเปิดกว้างสุด (ชดเชย delay ของ filter)"""
    bits, i_clean, _ = simulate_link(-10, 2000)
    best, best_open = 0, -np.inf
    for off in range(2 * sps):
        idx = np.arange(10, 1990) * sps + off
        s, b = i_clean[idx], bits[10:1990]
        opening = s[b == 1].min() - s[b == 0].max()
        if opening > best_open:
            best, best_open = off, opening
    return best

offset = find_sampling_offset()
bit_delay = offset // sps                   # filter delay in whole bits

def ber_monte_carlo(P_avg_dBm, n_bits=1_000_000):
    bits, _, i_rx = simulate_link(P_avg_dBm, n_bits)
    k = np.arange(20, n_bits - 2)           # skip filter start-up
    samples = i_rx[k * sps + offset]
    decided = samples > decision_threshold(P_avg_dBm)
    errors = np.count_nonzero(decided != (bits[k - bit_delay] == 1))
    return errors / k.size, errors

print(f"sampling offset = {offset} samples ({offset / sps:.2f} UI)")

# %% [markdown]
# ## 4. Eye diagram
# ซ้อนสัญญาณยาว 2 บิตหลายพันเส้นแล้วนับความหนาแน่น (เหมือนโหมด persistence ของสโคป)
# ซ้าย: แสงแรง ตาเปิดชัด / ขวา: แสงอ่อนใกล้ sensitivity noise ทำให้ตาเริ่มปิด

# %%
def plot_eye(ax, P_avg_dBm, n_bits=6000):
    _, _, i_rx = simulate_link(P_avg_dBm, n_bits)
    up = 8                                  # upsample so traces look continuous
    i_up = signal.resample_poly(i_rx, up, 1)
    n = sps * up                            # samples per bit after upsampling
    start = offset * up - n                 # sampling point sits at the centre of the window
    traces = np.array([i_up[k * n + start: k * n + start + 2 * n]
                       for k in range(20, n_bits - 3)]) * 1e6   # uA
    t = np.tile((np.arange(2 * n) - n) / (fs * up) * 1e12, traces.shape[0])   # ps
    cmap = LinearSegmentedColormap.from_list("eye", ["#dcebfb", BLUE, "#0b2f5c"])
    ax.hist2d(t, traces.ravel(), bins=[2 * n, 180], cmap=cmap, norm=LogNorm(), cmin=1)
    ax.axhline(decision_threshold(P_avg_dBm) * 1e6, color=ORANGE, lw=1.2, ls="--")
    ax.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax.set_title(f"Received power {P_avg_dBm:.0f} dBm  (Q = {q_factor(P_avg_dBm):.1f})")
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Photocurrent [µA]")
    ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
plot_eye(axes[0], -10)
plot_eye(axes[1], round(sensitivity_dBm) - 2)
fig.suptitle("NRZ eye diagram at 10 Gb/s (dashed line = decision threshold)")
fig.tight_layout()
fig.savefig("figures/eye_diagram.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. BER vs received power
# Monte Carlo วัด BER ได้ต่ำสุดราว 1e-5 (1 ล้านบิต) ส่วนที่ต่ำกว่านั้นใช้สูตรทฤษฎีต่อเส้น
# ถ้าจุด Monte Carlo ทับเส้นทฤษฎี แปลว่าโมเดลถูก
# ต่างกันไม่ถึง ~0.1 dB: filter Bessel มี overshoot เล็กน้อย พอสุ่มตัวอย่างที่จุดตาเปิดกว้างสุด
# สัญญาณจึงกว้างกว่า I1 - I0 นิดหน่อย ส่วน ISI ที่ bandwidth 0.75*Rb ยังน้อยมาก

# %%
p_theory = np.linspace(-26, -12, 300)
p_mc = np.arange(-26, -19.5, 0.5)
mc = [ber_monte_carlo(p) for p in p_mc]
ber_mc = np.array([m[0] for m in mc])
err_mc = np.array([m[1] for m in mc])
ok = err_mc >= 10                           # keep points with enough errors to trust

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.semilogy(p_theory, ber_theory(p_theory), color=BLUE, lw=2, label="Theory (Q-factor)")
ax.semilogy(p_mc[ok], ber_mc[ok], "o", color=ORANGE, ms=8,
            markeredgecolor="white", markeredgewidth=1.5, label="Monte Carlo (1M bits/point)")
ax.axhline(BER_target, color=GRAY, lw=0.8, ls="--")
ax.axvline(sensitivity_dBm, color=GRAY, lw=0.8, ls="--")
ax.annotate(f"Sensitivity {sensitivity_dBm:.1f} dBm\n@ BER {BER_target:.0e}",
            xy=(sensitivity_dBm, BER_target), xytext=(sensitivity_dBm + 0.6, 1e-9),
            color="#0b0b0b", fontsize=9)
ax.set_ylim(1e-14, 1)
ax.set_xlabel("Average received optical power [dBm]")
ax.set_ylabel("Bit error rate")
ax.set_title("BER vs received power — 10 Gb/s NRZ, PIN + TIA receiver")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("figures/ber_vs_power.png", dpi=150)
plt.show()

for p, b, e in zip(p_mc[ok], ber_mc[ok], err_mc[ok]):
    print(f"{p:6.1f} dBm   MC BER = {b:.2e} ({e:5d} errors)   theory = {ber_theory(p):.2e}")

# %% [markdown]
# ## 6. Link budget
# กำลังที่ตัวรับ = กำลังส่ง - loss ไฟเบอร์ x ระยะ - loss คอนเนกเตอร์
# ระยะสูงสุดคือจุดที่กำลังที่รับได้ยังสูงกว่า sensitivity + margin
# (โมเดลนี้ยังไม่รวม dispersion ซึ่งจำกัดระยะเพิ่มอีกที่ความเร็วสูง)

# %%
P_tx_dBm = -3.0         # transmitter average launch power [dBm]
fiber_loss = 0.4        # single-mode fiber loss at 1310 nm [dB/km]
connector_loss = 1.0    # 2 connectors x 0.5 dB
margin_dB = 3.0         # safety margin for aging/temperature

budget = P_tx_dBm - sensitivity_dBm
L_max = (budget - connector_loss - margin_dB) / fiber_loss
print(f"power budget  : {budget:.1f} dB")
print(f"max distance  : {L_max:.1f} km")

L = np.linspace(0, L_max * 1.4, 200)
P_rx = P_tx_dBm - connector_loss - fiber_loss * L
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(L, P_rx, color=BLUE, lw=2)
ax.axhline(sensitivity_dBm + margin_dB, color=ORANGE, lw=1.2, ls="--")
ax.axhline(sensitivity_dBm, color=GRAY, lw=0.8, ls=":")
ax.axvline(L_max, color=GRAY, lw=0.8, ls="--")
ax.text(0.5, sensitivity_dBm + margin_dB + 0.3, "Sensitivity + margin", color="#0b0b0b", fontsize=9)
ax.text(0.5, sensitivity_dBm + 0.3, "Sensitivity", color=GRAY, fontsize=9)
ax.text(L_max + 0.5, P_tx_dBm - 2, f"Max reach\n{L_max:.1f} km", color="#0b0b0b", fontsize=9)
ax.set_xlabel("Fiber length [km]")
ax.set_ylabel("Received power [dBm]")
ax.set_title("Link budget (attenuation-limited)")
fig.tight_layout()
fig.savefig("figures/link_budget.png", dpi=150)
plt.show()
