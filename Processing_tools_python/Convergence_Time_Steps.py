import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,                 
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 14.5,
    "font.size": 14.5,
    "legend.fontsize": 12.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
})
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import math

# load data
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

# path here
xlsx_path = r""
time_s, dx, dy, dz = lade_zeit_und_dipol(xlsx_path)

hbar_ev_s = 6.582119569e-16  # eV·s
dt = np.nanmean(np.diff(time_s))


N_freq = 4000
freqs_thz = np.linspace(0, 2000, N_freq)
omega_rad_s = freqs_thz * 1e12 * 2 * np.pi
energy_ev = omega_rad_s * hbar_ev_s

cutoffs = [20000, 22000, 23000, 24000, 24500, 25000]
labels  = ["20000 time steps", "22000 time steps", "23000 time steps",
           "24000 time steps", "24500 time steps", "25000 time steps"]

# remove dc offset
dx -= np.mean(dx); dy -= np.mean(dy); dz -= np.mean(dz)

debye_to_au = 1.0 / 2.541746
s_to_au     = 1.0 / 2.418884e-17
conversion_factor = debye_to_au * s_to_au

fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True, sharex=True, sharey=True)
axes = axes.flatten()

for i, (cutoff, label) in enumerate(zip(cutoffs, labels)):
    print(f"Berechne Spektrum für: {label}")

    ts = time_s[:cutoff]
    dx_cut = dx[:cutoff]; dy_cut = dy[:cutoff]; dz_cut = dz[:cutoff]


    window = np.cos(np.pi * (ts - ts[0]) / (2 * (ts[-1] - ts[0])))**2

    dx_w = dx_cut * window
    dy_w = dy_cut * window
    dz_w = dz_cut * window

   
    fourier_dx = np.zeros(N_freq, dtype=complex)
    fourier_dy = np.zeros(N_freq, dtype=complex)
    fourier_dz = np.zeros(N_freq, dtype=complex)
    for k, omega in enumerate(omega_rad_s):
        phase = np.exp(-1j * omega * ts)
        fourier_dx[k] = np.sum(dx_w * phase) * dt
        fourier_dy[k] = np.sum(dy_w * phase) * dt
        fourier_dz[k] = np.sum(dz_w * phase) * dt

   
    abs_dx = np.abs(fourier_dx) * conversion_factor
    abs_dy = np.abs(fourier_dy) * conversion_factor
    abs_dz = np.abs(fourier_dz) * conversion_factor
    power_spec = abs_dx**2 + abs_dy**2 + abs_dz**2
    if np.max(power_spec) > 0:
        power_spec = power_spec / np.max(power_spec)

 
    ax = axes[i]
    ax.plot(energy_ev, power_spec, color="black", linewidth=1.0)
    ax.set_xlim(0, 5)            
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())


    proxy = Line2D([], [], linestyle='None', label=label)
    ax.legend(handles=[proxy],
              loc='upper left',
              frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
              handlelength=0, handletextpad=0)

for ax in axes:
    ax.label_outer()


fig.supxlabel("Energy [eV]", y=0.03)
fig.supylabel("Dipole power [arb. units]", x=0.03)

plt.tight_layout()
plt.show()

