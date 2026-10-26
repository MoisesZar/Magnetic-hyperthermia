# Magnetic Hyperthermia: SAR for Different AC Waveforms

Micromagnetic simulation code to compute the **Specific Absorption Rate (SAR)**
of assemblies of magnetic nanoparticles under different AC field waveforms.

Supported waveforms:
- Sinusoidal
- Triangular
- Trapezoidal
- Square

## Physical model

Each simulation cell represents a nanoparticle with:
- **Uniaxial anisotropy** (random easy axis per cell)
- An **external AC field** with a selectable waveform

> ⚠️ **No dipolar interactions** are included in this model.
> The assembly is treated as a collection of non-interacting particles.
MoisesZar
moisesg.zarzoza@uclm.es

## Requirements

- Python 3.9+
- [Ubermag](https://ubermag.github.io/) (`oommfc`, `discretisedfield`, `micromagneticmodel`)
- OOMMF installed and available in `PATH`
- `numpy`, `scipy`, `matplotlib`

```bash
pip install oommfc discretisedfield micromagneticmodel numpy scipy matplotlib
