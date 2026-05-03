"""
common/config.py — Parameter global DLA accelerator.

Semua anggota mengacu ke file ini supaya parameter konsisten.
Ubah di sini, semua modul ikut berubah.
"""


# === PE Array ===
PE_ARRAY_ROWS = 8           # jumlah PE baris (M)
PE_ARRAY_COLS = 8           # jumlah PE kolom (N)
CLOCK_FREQ_MHZ = 200        # frekuensi clock target (MHz)

# === Precision ===
DEFAULT_PRECISION = 8       # bit-width default (INT8)
ACCUMULATOR_WIDTH = 32      # accumulator bit-width
SUPPORTED_PRECISIONS = [4, 8, 16, 32]  # bit-width yang disimulasikan

# === Memory hierarchy ===
SRAM_SIZE_KB = 64           # on-chip SRAM total (KB)
SRAM_BANKS = 4              # jumlah SRAM bank
SRAM_READ_LATENCY = 1       # cycles
SRAM_WRITE_LATENCY = 1      # cycles
DRAM_READ_LATENCY = 100     # cycles (off-chip, mahal!)
DRAM_BANDWIDTH_GBs = 8      # GB/s DRAM bandwidth

# === Energy model (estimasi 180nm GF180MCU) ===
ENERGY_MAC_INT8_PJ = 0.2    # pJ per INT8 MAC operation
ENERGY_MAC_INT16_PJ = 0.8   # pJ per INT16 MAC
ENERGY_MAC_FP32_PJ = 3.7    # pJ per FP32 MAC
ENERGY_SRAM_READ_PJ = 5.0   # pJ per SRAM read (64KB)
ENERGY_SRAM_WRITE_PJ = 5.0  # pJ per SRAM write
ENERGY_DRAM_READ_PJ = 200   # pJ per DRAM access (off-chip)

# === CNN workload: LeNet-5 layer specs ===
# Format: (name, C_in, H_in, W_in, C_out, K, stride, padding)
LENET5_LAYERS = [
    ("conv1",  1, 28, 28,  6, 5, 1, 0),   # 28x28x1  → 24x24x6
    ("conv2",  6, 12, 12, 16, 5, 1, 0),   # 12x12x6  → 8x8x16
    ("fc1",  256,  1,  1, 120, 1, 1, 0),   # 256 → 120 (1x1 conv)
    ("fc2",  120,  1,  1,  84, 1, 1, 0),   # 120 → 84
    ("fc3",   84,  1,  1,  10, 1, 1, 0),   # 84 → 10
]

# === Plotting ===
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)
