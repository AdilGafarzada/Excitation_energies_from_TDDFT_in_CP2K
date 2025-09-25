import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,                 
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator


def lade_zeit_und_dipol(xlsx_path, dt_fs=0.02):
    df = pd.read_excel(xlsx_path, header=None)
    full_text_rows = df.astype(str).apply(lambda row: ' '.join(row.values), axis=1)
    parsed = pd.DataFrame({
        "x": full_text_rows.str.extract(r"X=\s*([-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?)")[0],
        "y": full_text_rows.str.extract(r"Y=\s*([-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?)")[0],
        "z": full_text_rows.str.extract(r"Z=\s*([-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?)")[0],
    })
    parsed = parsed.dropna()
    N = len(parsed)
    time_s = np.arange(N) * dt_fs * 1e-15
    return time_s, parsed["x"].astype(float).values, parsed["y"].astype(float).values, parsed["z"].astype(float).values

# normalize & log-scale
use_log = False    
normalize = False

# load data 
xlsx_path = r""
time_s, dx, dy, dz = lade_zeit_und_dipol(xlsx_path)

# Konstanten
hbar_ev_s = 6.582119569e-16
debye_to_au = 0.393456
s_to_au = 1 / 2.418884e-17

dx -= np.mean(dx)
dy -= np.mean(dy)
dz -= np.mean(dz)

dt = np.nanmean(np.diff(time_s))
window = np.cos(np.pi * (time_s - time_s[0]) / (2 * (time_s[-1] - time_s[0])))**2

dx_w = dx * window
dy_w = dy * window
dz_w = dz * window

N_freq = 4000
freqs_thz = np.linspace(0, 2000, N_freq)
omega_rad_s = freqs_thz * 1e12 * 2 * np.pi
energy_ev = omega_rad_s * hbar_ev_s

fourier_dx = np.zeros(N_freq, dtype=complex)
fourier_dy = np.zeros(N_freq, dtype=complex)
fourier_dz = np.zeros(N_freq, dtype=complex)

for k, omega in enumerate(omega_rad_s):
    exp_phase = np.exp(-1j * omega * time_s)
    fourier_dx[k] = np.sum(dx_w * exp_phase) * dt
    fourier_dy[k] = np.sum(dy_w * exp_phase) * dt
    fourier_dz[k] = np.sum(dz_w * exp_phase) * dt

conversion_factor = debye_to_au * s_to_au
abs_dx = np.abs(fourier_dx) * conversion_factor
abs_dy = np.abs(fourier_dy) * conversion_factor
abs_dz = np.abs(fourier_dz) * conversion_factor

power_spec = abs_dx**2 + abs_dy**2 + abs_dz**2

# Normierung
if normalize:
    dipole_power = power_spec / np.max(power_spec)
else:
    dipole_power = power_spec

# Format y-Achse (bei log)
class MixedLogFormatter(ticker.LogFormatter):
    def __call__(self, x, pos=None):
        if x >= 1e-4:
            return f"{x:.4g}"
        else:
            return f"$10^{{{int(np.log10(x))}}}$"

# Plot 
fig, ax_energy = plt.subplots(figsize=(8, 5))

if use_log:
    ax_energy.set_yscale("log")

ax_energy.plot(energy_ev, dipole_power, color='black')
ax_energy.set_xlabel("Energy [eV]")
ax_energy.set_ylabel("Dipole power [arb. units]")

if use_log:
    ax_energy.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax_energy.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10))
    ax_energy.yaxis.set_major_formatter(MixedLogFormatter())
    ax_energy.grid(True, which='major', linestyle='--', linewidth=0.5)
else:
    ax_energy.yaxis.set_major_locator(ticker.AutoLocator())
    ax_energy.yaxis.set_minor_locator(AutoMinorLocator())
    ax_energy.grid(True, which='both', linestyle='--', linewidth=0.5)

ax_energy.set_xlim(0, 8)

def energy_to_freq(ev): return ev / hbar_ev_s / (2 * np.pi * 1e12)
def freq_to_energy(thz): return thz * 2 * np.pi * 1e12 * hbar_ev_s

#ax_freq = ax_energy.secondary_xaxis('top', functions=(energy_to_freq, freq_to_energy))
#ax_freq.set_xlabel("Frequency [THz]")
#ax_freq.set_xlim(energy_to_freq(0), energy_to_freq(8))

plt.tight_layout()

plt.show()
