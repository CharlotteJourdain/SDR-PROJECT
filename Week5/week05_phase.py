import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Receiver parameters
SAMPLE_RATE = 20e6
CHANNEL = 36
SYNC_THRESHOLD = 0.56
SYNC_LENGTH = 320
DELAY = 320
NS = 8

# Read the FFT output recorded from GNU Radio
v = np.fromfile("week05_fft.bin", np.complex64)
v = v.reshape(-1, 64).astype(np.complex128)

# IEEE 802.11 pilot polarity sequence
def scrambler(state, n):
    out = np.zeros(n, dtype=np.int8)

    for i in range(n):
        bit = ((state >> 6) ^ (state >> 3)) & 1
        state = ((state << 1) | bit) & 0x7F
        out[i] = bit

    return out

POL = 1 - 2 * scrambler(0x7F, 127)

# Pilot and DATA subcarriers
PIL_K = np.array([-21, -7, 7, 21])
PIL_V = np.array([1, 1, 1, -1])

DATA_K = np.array([
    k for k in range(-26, 27)
    if k != 0 and k not in PIL_K
])

IDX_PIL = PIL_K + 32
IDX_DATA = DATA_K + 32

# Known Long Training Sequence
L = np.array([
     1,  1, -1, -1,  1,  1, -1,  1,
    -1,  1,  1,  1,  1,  1,  1, -1,
    -1,  1,  1, -1,  1, -1,  1,  1,
     1,  1,  0,  1, -1, -1,  1,  1,
    -1,  1, -1,  1, -1, -1, -1, -1,
    -1,  1,  1, -1, -1,  1, -1,  1,
    -1,  1,  1,  1,  1
])

LFULL = np.zeros(64)
LFULL[np.arange(-26, 27) + 32] = L

# Detect active OFDM bursts
used = [i for i in range(6, 59) if i != 32]

energy = np.sum(np.abs(v[:, used]) ** 2, axis=1)
energy_threshold = 0.2 * np.percentile(energy, 99)

mask = energy > energy_threshold

starts = np.flatnonzero(
    mask & ~np.r_[False, mask[:-1]]
)

ends = np.flatnonzero(
    mask & ~np.r_[mask[1:], False]
)

lengths = ends - starts + 1

starts = starts[lengths >= 3 + NS]
starts = starts[starts + 3 + NS <= len(v)]

print("FFT vectors:", len(v))
print("Detected bursts:", len(starts))

if len(starts) == 0:
    raise SystemExit("No frames detected.")

# Extract the two LTS vectors, SIGNAL and DATA symbols
idx = starts[:, None] + np.arange(3 + NS)[None, :]
V = v[idx]

# Channel estimation from the two LTS symbols
H = ((V[:, 0] + V[:, 1]) / 2) * LFULL

H[:, LFULL == 0] = 1.0
H[np.abs(H) < 1e-9] = 1.0

# Channel equalization
X = V[:, 2:] / H[:, None, :]

# Estimate common phase error from the four pilots
polarity = POL[np.arange(NS + 1) % 127]

pilots = (
    X[:, :, IDX_PIL]
    * PIL_V
    * polarity[None, :, None]
)

phase = np.angle(np.sum(pilots, axis=-1))

# DATA only: exclude SIGNAL symbol
data_before = X[:, 1:, IDX_DATA]

# Correct the phase of every DATA subcarrier
data_after = (
    data_before
    * np.exp(-1j * phase[:, 1:, None])
)

# Select correctly aligned frames
def frame_error(x):
    ref = np.where(x.real >= 0, 1.0, -1.0)
    return np.mean(np.abs(x - ref) ** 2, axis=(-1, -2))

quality = frame_error(data_after)
ok = quality < 0.5

n_frames = np.sum(ok)

print("Accepted frames:", n_frames, "/", len(ok))

if n_frames == 0:
    raise SystemExit("No correctly aligned frames.")

before = data_before[ok].ravel()
after = data_after[ok].ravel()

# RMS EVM relative to ideal BPSK decisions
def rms_evm(x):
    ref = np.where(x.real >= 0, 1.0, -1.0)

    return np.sqrt(
        np.sum(np.abs(x - ref) ** 2)
        / np.sum(np.abs(ref) ** 2)
    )

evm_before = rms_evm(before)
evm_after = rms_evm(after)

print(f"EVM before phase correction: {evm_before:.4f}")
print(f"EVM after phase correction : {evm_after:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

datasets = [
    (
        before,
        f"Before phase correction\nEVM = {evm_before:.3f}"
    ),
    (
        after,
        f"After pilot-based phase correction\nEVM = {evm_after:.3f}"
    )
]

for ax, (data, title) in zip(axes, datasets):

    ax.hist2d(
        data.real,
        data.imag,
        bins=120,
        range=[[-2, 2], [-2, 2]],
        norm=LogNorm()
    )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)

    ax.grid(alpha=0.2)

fig.suptitle(
    "Week 05 - Phase Offset Correction\n"
    "DATA constellation before and after pilot-based correction",
    fontsize=14
)

parameter_text = (
    "Input: Sample1_20MHz_Channel36.bin   |   "
    "Sample rate: 20 MHz   |   Channel: 36\n"
    "Sync Short threshold: 0.56   |   "
    "Sync Long: 320   |   Delay: 320   |   "
    "Pilots: {-21, -7, 7, 21}\n"
    f"Accepted frames: {n_frames}   |   "
    f"DATA symbols/frame: {NS}"
)

fig.text(
    0.5,
    0.02,
    parameter_text,
    ha="center",
    va="bottom",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.11, 1, 0.91])

plt.savefig(
    "week05_phase_comparison.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()
