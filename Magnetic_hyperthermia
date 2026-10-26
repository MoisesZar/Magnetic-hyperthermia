
---

## 3. Código limpio (solo simulación)

```python
#!/usr/bin/env python
# coding: utf-8
"""
Magnetic hyperthermia: Specific Absorption Rate (SAR) of nanoparticle
assemblies under different AC signal waveforms.

Simulates the time-dependent magnetization of an assembly of magnetic
nanoparticles under different AC field waveforms (sinusoidal, triangular,
trapezoidal, square) and computes the SAR.

⚠️ Important:
  - No dipolar interactions are included in this model.
  - The damping alpha enters the thermal evolver. Very small values give
    excessive noise. Recommended: alpha = 0.5. Do not go below ~0.1.
  - The excitation frequency should be in the MHz range (e.g. 1 MHz).

Built on Ubermag / OOMMF.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.fft import fft, ifft, fftfreq

import oommfc as mc
import discretisedfield as df
import micromagneticmodel as mm


# ----------------------------------------------------------------------
# Physical and numerical parameters
# ----------------------------------------------------------------------
n0 = 0.8388e-9                  # reference length (m)
rho = 5170                      # density (kg/m^3)
MW = 234.37                     # molar mass (g/mol)
Na = 6.022e23                   # Avogadro constant
n = Na * rho * 1000 / MW        # number density (1/m^3)
Ms = 4 * mm.consts.muB * n      # saturation magnetization (A/m)
K = 1.1e4                       # uniaxial anisotropy constant (J/m^3)

# ⚠️ alpha enters the thermal evolver. Very small values give noise.
# Recommended: 0.5. Do not go below ~0.1.
alpha = 0.5

# ⚠️ Excitation frequency must be in the MHz range.
f = 1e6                         # Hz

# Simulation defaults
D = 15                          # particle diameter in units of n0
temp = 310                      # temperature (K)
dt = 1e-12                      # timestep (s)


# ----------------------------------------------------------------------
# AC waveforms
# These are called by Ubermag's Zeeman term as func(t) (scalar t).
# They use the global frequency f.
# ----------------------------------------------------------------------
def sin_wave(t):
    """Sinusoidal waveform, range [-1, 1]."""
    return np.sin(2 * np.pi * f * t)


def square_wave(t):
    """Square waveform: -1 for the first half-cycle, +1 for the second."""
    return -1.0 if t < 1 / (2 * f) else 1.0


def triangle_wave(t):
    """Triangular waveform, range [-1, 1]."""
    if t < 1 / (2 * f):
        return 4 * f * t - 1
    return -4 * f * t + 3


def trapezoid_wave(t):
    """Trapezoidal waveform, range [-1, 1]."""
    if t <= 1 / (6 * f):
        return -1
    elif 1 / (6 * f) < t <= 1 / (2 * f):
        return 6 * t * f - 2
    elif 1 / (2 * f) < t <= 2 / (3 * f):
        return 1
    else:
        return -6 * t * f + 5


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
def simulate(D, f, H0, waveform, dt, Temp):
    """
    Run one OOMMF simulation for a given waveform and field amplitude.

    Parameters
    ----------
    D : int
        Particle diameter in units of n0.
    f : float
        Excitation frequency (Hz).
    H0 : float
        Field amplitude (A/m).
    waveform : callable
        Waveform function of time, range [-1, 1].
    dt : float
        Timestep (s).
    Temp : float
        Temperature (K).

    Returns
    -------
    t : ndarray
        Time array.
    M : ndarray
        Magnetization along z (A/m).
    B : ndarray
        AC field along z (T).
    """
    dNP = D * n0
    cell = (dNP, dNP, dNP)
    d = 15 * dNP
    region = df.Region(p1=(0, 0, 0), p2=(d, d, d))
    mesh = df.Mesh(region=region, cell=cell)

    system = mm.System(name='Hyperthermia')
    system.m = df.Field(
        mesh,
        nvdim=3,
        value=lambda pos: np.random.uniform(-1, 1, size=3),
        norm=Ms,
        valid="norm",
    )

    # Random uniaxial easy axis per cell
    u = df.Field(
        mesh, nvdim=3,
        value=lambda pos: np.random.uniform(-1, 1, size=3),
    )

    # NOTE: dipolar interactions are NOT included in this model
    system.energy = (
        mm.UniaxialAnisotropy(K=K, u=u)
        + mm.Zeeman(H=(0, 0, H0), func=waveform, dt=dt, t0=-dt)
    )
    system.dynamics = (
        mm.Damping(alpha=alpha)
        + mm.Precession(gamma0=mm.consts.gamma0)
    )
    system.T = Temp

    evolver = mc.UHH_ThetaEvolver(fixed_timestep=dt, alpha=alpha)
    td = mc.TimeDriver(evolver=evolver)
    td.drive(system, t=1 / f, n=100)

    t = system.table.data['t'].values
    M = Ms * system.table.data['mz'].values
    B = system.table.data['Bz_zeeman'].values / 1000    # T
    return t, M, B


# ----------------------------------------------------------------------
# SAR calculation
# ----------------------------------------------------------------------
def compute_SAR(t, M, B, f, rho=rho):
    """
    Compute the Specific Absorption Rate from the time-dependent
    magnetization.

    Returns
    -------
    SAR_raw : float
        SAR from the raw magnetization (W/g).
    SAR_filtered : float
        SAR after a band-pass FFT filter around f (W/g).
    """
    # Raw
    dM_dt = np.gradient(M, t)
    integral = simpson(B * dM_dt, x=t)
    SAR_raw = f * integral / (rho * 1000)

    # Band-pass filter around the excitation frequency
    N = len(t)
    M_f = fft(M)
    freqs = fftfreq(N, t[1] - t[0])
    bandwidth = 0.1 * f
    mask = (np.abs(freqs) > (f - bandwidth)) & (np.abs(freqs) < (f + bandwidth))
    M_f_filtered = np.zeros_like(M_f)
    M_f_filtered[mask] = M_f[mask]
    M_filtered = ifft(M_f_filtered).real

    dM_dt_filtered = np.gradient(M_filtered, t)
    integral_filtered = simpson(B * dM_dt_filtered, x=t)
    SAR_filtered = f * integral_filtered / (rho * 1000)

    return SAR_raw, SAR_filtered


# ----------------------------------------------------------------------
# Field sweep
# ----------------------------------------------------------------------
def field_sweep(D, f, temp, waveform, waveform_name,
                H_min=0.5, H_max=5.0, H_step=0.5, dt=1e-12):
    """
    Sweep the AC field amplitude and compute SAR for each value.

    Returns
    -------
    fields : ndarray
        Field amplitudes (mT).
    SAR_raw : ndarray
        Raw SAR values (W/g).
    SAR_filtered : ndarray
        Filtered SAR values (W/g).
    """
    fields = np.arange(H_min, H_max + H_step / 2, H_step)
    SAR_raw_list = []
    SAR_filt_list = []

    for H_mT in fields:
        H0 = H_mT * 1e-3 / mm.consts.mu0
        t, M, B = simulate(D, f, H0, waveform, dt, temp)
        sar_raw, sar_filt = compute_SAR(t, M, B, f)
        SAR_raw_list.append(sar_raw)
        SAR_filt_list.append(sar_filt)
        print(
            f"[{waveform_name:12s}] H = {H_mT:5.2f} mT | "
            f"SAR = {sar_raw:10.4e} W/g | "
            f"SAR filtered = {sar_filt:10.4e} W/g"
        )

    return fields, np.array(SAR_raw_list), np.array(SAR_filt_list)


# ----------------------------------------------------------------------
# Optional: preview a waveform
# ----------------------------------------------------------------------
def plot_waveform(waveform, name, f=f):
    """Plot one cycle of a waveform."""
    t = np.linspace(0, 1 / f, 1000)
    y = np.array([waveform(ti) for ti in t])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(f * t, y)
    ax.set_xlabel(r'$f \cdot t$')
    ax.set_ylabel('H / H0')
    ax.set_title(name)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    waveforms = [
        (sin_wave, "Sinusoidal"),
        (triangle_wave, "Triangular"),
        (trapezoid_wave, "Trapezoidal"),
        (square_wave, "Square"),
    ]

    # Run simulations for each waveform
    results = {}
    for func, name in waveforms:
        fields, sar_raw, sar_filt = field_sweep(
            D=D, f=f, temp=temp,
            waveform=func, waveform_name=name,
            H_min=0.5, H_max=5.0, H_step=0.5, dt=dt,
        )
        results[name] = (fields, sar_raw, sar_filt)

    # Plot SAR vs field amplitude
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {
        'Sinusoidal': 'o',
        'Triangular': 's',
        'Trapezoidal': '^',
        'Square': 'D',
    }
    for name, (fields, sar_raw, _) in results.items():
        ax.plot(fields, sar_raw, marker=markers[name], label=name)
    ax.set_xlabel(r'$\mu_0 H$ (mT)')
    ax.set_ylabel('SAR (W/g)')
    ax.set_title(
        f'SAR vs field amplitude  (f = {f/1e6:.1f} MHz, T = {temp} K)'
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Save results
    for name, (fields, sar_raw, sar_filt) in results.items():
        data = np.column_stack((fields, sar_raw, sar_filt))
        fname = f'SAR_{name.lower()}_f_{f/1e6:.1f}MHz_D_{D}nm_T_{temp}K.txt'
        np.savetxt(
            fname,
            data,
            fmt='%.5e',
            header='H (mT)  SAR_raw (W/g)  SAR_filtered (W/g)',
            delimiter=' ',
        )
        print(f"Saved {fname}")

    # Optional: preview the waveforms
    # for func, name in waveforms:
    #     plot_waveform(func, name)
