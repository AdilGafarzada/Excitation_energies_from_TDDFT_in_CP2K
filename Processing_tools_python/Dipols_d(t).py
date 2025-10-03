import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,                 
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator


def lade_zeit_und_dipol(xlsx_path, dt_fs=0.02):
    """
    Läd eine Excel-Datei mit TDDFT-Dipoldaten und extrahiert Zeit und d_x, d_y, d_z.
    """
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

    dipole_x = parsed["x"].astype(float)
    dipole_y = parsed["y"].astype(float)
    dipole_z = parsed["z"].astype(float)

    return time_s, dipole_x.values, dipole_y.values, dipole_z.values


xlsx_path = r"Data\Dipole_outputs\Na48 Distance Test 3-14 A\33x33x55\10\Microsoft Excel-Arbeitsblatt (neu).xlsx"
time_s, dx, dy, dz = lade_zeit_und_dipol(xlsx_path)
time_fs = time_s * 1e15  
# Remove DC offset
dx -= np.mean(dx)
dy -= np.mean(dy)
dz -= np.mean(dz)

# Debye to atomic units 
dx *= 0.393456
dy *= 0.393456
dz *= 0.393456

components = {
    r"$d_x$": dx,
    r"$d_y$": dy,
    r"$d_z$": dz,
}


fig1, axes1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for ax, (label, data) in zip(axes1, [(r"$d_x$", dx), (r"$d_y$", dy)]):
    ax.plot(time_fs, data)
    ax.set_ylabel(f"{label} [atomic units]")
    ax.set_title(f"{label}(t)")
    ax.grid(True)
axes1[-1].set_xlabel("Time [fs]")
plt.tight_layout()
plt.show()

# Figure 2: d_z separate
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3.2))
ax2.plot(time_fs, dz)
ax2.set_ylabel(r"$d_z$ [atomic units]")
ax2.set_title(r"$d_z$(t)")
ax2.grid(True)
ax2.set_xlabel("Time [fs]")
plt.tight_layout()
plt.show()
