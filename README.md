# AC–DC Rectifier & Signal Conditioning Lab

A production-ready Streamlit dashboard that models an AC source, silicon-diode rectifier, capacitor-input smoothing filter, and active op-amp-style signal conditioning stage for a microcontroller ADC.

## Setup

```bash
cd /Users/yousif
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The engine samples at 100 kHz across four source cycles. The first cycle is excluded from steady-state metrics so capacitor startup does not distort the reported ripple.

## Circuit model

The source is (v_{in}(t)=\sqrt{2}V_{rms}\sin(2\pi ft)). For a bridge, the raw rectified signal is (\max(|v_{in}|-2V_f,0)); for half-wave it is (\max(v_{in}-V_f,0)).

At each sample, the capacitor charges instantly to the rectified voltage when the diode is forward-biased. Otherwise it discharges through the load:

\[
v_{out}[n]=v_{out}[n-1]\exp\left(-\frac{\Delta t}{R_LC}\right)
\]

Reported metrics use the post-warm-up output:

- (V_{dc}=\frac{1}{T}\int v_{out}(t)dt)
- (V_{rms}=\sqrt{\frac{1}{T}\int v_{out}^2(t)dt})
- (V_{r(pp)}=\max(v_{out})-\min(v_{out}))
- (r=\sqrt{(V_{rms}/V_{dc})^2-1}\times100\%)

The conditioning stage scales the filtered waveform to 95% of the selected ADC maximum, applies a first-order Butterworth low-pass filter through `scipy.signal`, and clips the result to the safe ADC range. The FFT uses `numpy.fft.rfft` and returns a one-sided amplitude spectrum.
