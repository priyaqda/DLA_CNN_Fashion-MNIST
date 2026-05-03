# DLA CNN Accelerator — Level 1: Modeling & Simulasi
## LeNet-5 · Fashion-MNIST · 8×8 PE Array · INT8 · Weight-Stationary · GF180MCU 180nm

---

## Struktur Direktori Lengkap

```
dla_level1/
│
│   run_pipeline.py             ← ★ JALANKAN INI untuk full pipeline
│   README.md                   ← file ini
│
├── common/                     ─── Shared: dipakai semua anggota
│   ├── config.py               # Parameter global chip (array size, clock, energi, dll)
│   ├── interfaces.py           # Dataclass kontrak I/O antar modul
│   ├── utils.py                # Helper: print_table, plot_bar, plot_heatmap
│   └── __init__.py
│
├── anggota1_pe_array/          ─── Anggota 1: PE Array Model
│   ├── mac_unit.py             # Model 1 MAC unit (acc += w × a)
│   ├── pe_array.py             # Model 8×8 PE array, profiling cycles & utilization
│   ├── conv_engine.py          # Conv2D engine: jalankan multi-layer CNN
│   ├── inference.py            # ★ Inference dengan weights terlatih (FP32 vs INT8)
│   ├── test_pe_array.py        # Unit test: correctness + profiling sweep
│   └── __init__.py
│
├── anggota2_dataflow/          ─── Anggota 2: Dataflow Simulation
│   ├── weight_stationary.py    # Simulasi Weight-Stationary dataflow
│   ├── output_stationary.py    # Simulasi Output-Stationary dataflow
│   ├── dataflow_analyzer.py    # Perbandingan WS vs OS: DRAM traffic, energy, BW
│   ├── test_dataflow.py        # Test + tabel perbandingan
│   └── __init__.py
│
├── anggota3_buffer_tiling/     ─── Anggota 3: Buffer & Tiling
│   ├── sram_model.py           # Model SRAM 64KB: partisi, latency, energy, area
│   ├── tiling_engine.py        # Optimizer tile size → minimize DRAM traffic
│   ├── memory_hierarchy.py     # Energy model DRAM↔SRAM↔PE + roofline analysis
│   ├── test_tiling.py          # Test: optimal tile sweep per layer
│   └── __init__.py
│
├── anggota4_quantization/      ─── Anggota 4: Quantization
│   ├── quantizer.py            # FP32 → INT8 symmetric + per-channel quantizer
│   ├── accuracy_analysis.py    # Ukur MSE & cosine similarity FP32 vs INT8
│   ├── hw_cost_model.py        # Estimasi area & power vs bit-width (180nm)
│   ├── train_lenet.py          # ★ Training LeNet-5 pada Fashion-MNIST
│   ├── export_weights.py       # ★ Quantize FP32→INT8 + export HEX untuk Verilog
│   ├── test_quantization.py    # Test: quantizer, accuracy, cost model
│   └── __init__.py
│
├── anggota5_integration/       ─── Anggota 5: Integration & Benchmark
│   ├── dla_simulator.py        # End-to-end DLA simulator (gabungkan semua modul)
│   ├── benchmark.py            # Sweep: array size, precision, dataflow
│   ├── gpu_baseline.py         # Perbandingan DLA vs Mobile GPU vs Desktop GPU
│   ├── run_demo.py             # Demo benchmark lengkap (6 analisis)
│   └── __init__.py
│
├── weights/                    ─── Output training (auto-generated, jangan diedit manual)
│   ├── lenet5_fp32.npy         # Weights FP32 hasil training (referensi akurasi)
│   ├── lenet5_int8.npy         # Weights INT8 + scale factors (untuk Python sim)
│   ├── lenet5_int8.hex         # ★ Weights HEX untuk Verilog $readmemh (Level 2)
│   ├── lenet5_scales.hex       # Scale factors per-channel (untuk dequant di RTL)
│   ├── weight_summary.txt      # Ringkasan: shape, range, MSE tiap layer
│   └── accuracy.txt            # Log akurasi training per epoch
│
├── data/                       ─── Dataset (auto-download saat training, jangan diedit)
│   └── FashionMNIST/
│       └── raw/                # File binary Fashion-MNIST (~30 MB, 4 file)
│
└── docs/
    └── interface_spec.md       # Kontrak I/O antar subsistem (→ port list Verilog Level 2)
```

---

## Cara Menjalankan

### ★ Full Pipeline — satu perintah (direkomendasikan)

```bash
# Step 0: Install dependencies (sekali saja di laptop)
pip install torch torchvision numpy matplotlib

# Step 1: Masuk ke folder proyek
cd dla_level1

# Step 2: Jalankan pipeline
python run_pipeline.py
```

Pipeline otomatis menjalankan 4 step berurutan:

```
Step 1  Training LeNet-5 pada Fashion-MNIST   →  weights/lenet5_fp32.npy
        (dataset ~30 MB didownload otomatis)

Step 2  Quantize FP32 → INT8 + export HEX    →  weights/lenet5_int8.hex
        (siap untuk Verilog $readmemh)

Step 3  Verifikasi inference FP32 vs INT8     →  accuracy report di terminal

Step 4  DLA benchmark end-to-end              →  performance report di terminal
```

Estimasi waktu total: **10–20 menit** (sebagian besar untuk training di Step 1).

---

### Menjalankan per step secara manual

Kalau ingin jalankan satu step saja (misalnya sudah punya weights dan hanya mau benchmark):

```bash
# Training saja
python -m anggota4_quantization.train_lenet

# Export saja (butuh lenet5_fp32.npy dulu)
python -m anggota4_quantization.export_weights

# Inference verification saja (butuh kedua weights)
python -m anggota1_pe_array.inference

# Benchmark DLA saja (tidak butuh weights)
python -m anggota5_integration.run_demo
```

### Menjalankan test per anggota

```bash
python -m anggota1_pe_array.test_pe_array
python -m anggota2_dataflow.test_dataflow
python -m anggota3_buffer_tiling.test_tiling
python -m anggota4_quantization.test_quantization
```

---

## Urutan dependency antar file

```
train_lenet.py
    └──→ weights/lenet5_fp32.npy
              └──→ export_weights.py
                        ├──→ weights/lenet5_int8.npy
                        │         └──→ inference.py (verifikasi)
                        └──→ weights/lenet5_int8.hex
                                  └──→ Level 2 Verilog ROM

pe_array.py      →  PEArrayResult    →  dataflow, dla_simulator
dataflow/*.py    →  DataflowResult   →  tiling, dla_simulator
tiling/*.py      →  TilingResult     →  dla_simulator
hw_cost_model.py →  area & power     →  dla_simulator
dla_simulator.py →  DLABenchmark     →  benchmark, run_demo
```

---

## Koneksi ke Level 2 (Verilog)

File `weights/lenet5_int8.hex` yang dihasilkan pipeline siap dipakai di Verilog:

```verilog
parameter TOTAL_BYTES = 45134;
reg [7:0] w_rom [0:TOTAL_BYTES-1];
initial $readmemh("weights/lenet5_int8.hex", w_rom);
```

Address map tiap layer ada di header file `lenet5_int8.hex` dan di `weights/weight_summary.txt`.

---

## Dependencies

| Package | Dibutuhkan untuk | Install |
|---------|-----------------|---------|
| `numpy` | Semua modul simulasi | `pip install numpy` |
| `matplotlib` | Plotting grafik | `pip install matplotlib` |
| `torch` | Training LeNet-5 (Step 1) | `pip install torch` |
| `torchvision` | Download Fashion-MNIST | `pip install torchvision` |

Minimum tanpa training: `pip install numpy matplotlib`
Full pipeline: `pip install torch torchvision numpy matplotlib`

---

## .gitignore yang disarankan

```
__pycache__/
*.pyc
*.pyo
data/
weights/*.npy
```

File `weights/lenet5_int8.hex` boleh di-commit karena ukurannya kecil (~45 KB)
dan langsung dibutuhkan di Level 2. File `.npy` lebih besar dan bisa di-regenerate
dengan menjalankan pipeline ulang.

---

## Interface antar subsistem

Lihat `docs/interface_spec.md` untuk kontrak I/O lengkap.
Setiap field di dataclass `common/interfaces.py` akan menjadi **port sinyal Verilog** di Level 2.
