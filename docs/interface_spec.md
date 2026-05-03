# Interface Specification — Kontrak Antar Subsistem

## Tujuan Dokumen

Dokumen ini mendefinisikan **kontrak I/O** antar modul.
Setiap anggota WAJIB mengikuti interface yang sudah ditentukan
di `common/interfaces.py`. Perubahan interface harus disetujui semua anggota.

> **Catatan untuk Level 2 (RTL):**  
> Setiap field di dataclass ini akan menjadi port/signal di Verilog.
> Desain interface sekarang = mendesain port list hardware nanti.

---

## Peta Ketergantungan

```
Anggota 1 (PE Array)
    │
    ├──→ Anggota 2 (Dataflow)   : PEArrayResult.total_cycles
    │         │
    │         └──→ Anggota 3 (Buffer) : DataflowResult.total_dram_reads
    │                   │
    │                   └──→ Anggota 5 (Integration) : TilingResult
    │
    ├──→ Anggota 4 (Quantization) : bit_width → PE area/power model
    │         │
    │         └──→ Anggota 5 : QuantizationResult
    │
    └──→ Anggota 5 (Integration) : PEArrayResult
```

---

## Interface Detail

### 1. PE Array → Dataflow (Anggota 1 → 2)

```python
PEArrayResult:
    output_feature_map: np.ndarray  # computed output
    total_macs: int                 # jumlah MAC operations
    total_cycles: int               # estimated clock cycles
    utilization: float              # 0.0 - 1.0
    pe_rows: int
    pe_cols: int
```

**Kapan dipakai:** Anggota 2 menerima `total_cycles` dan `pe_rows × pe_cols`
untuk menentukan apakah layer tersebut compute-bound atau memory-bound.

### 2. Dataflow → Buffer (Anggota 2 → 3)

```python
DataflowResult:
    strategy_name: str              # "weight_stationary" / "output_stationary"
    total_dram_reads: int           # dipakai Anggota 3 untuk optimasi tiling
    total_sram_reads: int
    total_sram_writes: int
    data_reuse_factor: float
    bandwidth_required_GBs: float
    cycle_count: int
```

**Kapan dipakai:** Anggota 3 menggunakan `total_dram_reads` untuk memvalidasi
apakah tiling strategy berhasil mengurangi DRAM access.

### 3. Buffer/Tiling → Integration (Anggota 3 → 5)

```python
TilingResult:
    tile_h, tile_w, tile_c: int     # optimal tile dimensions
    num_tiles: int
    sram_hit_rate: float            # 0.0 - 1.0
    dram_traffic_bytes: int
    optimal: bool                   # apakah muat di SRAM
```

### 4. Quantization → Integration (Anggota 4 → 5)

```python
QuantizationResult:
    bit_width: int
    accuracy_fp32: float
    accuracy_quantized: float
    accuracy_drop: float            # penurunan accuracy (%)
    area_reduction_factor: float    # vs FP32
    energy_reduction_factor: float  # vs FP32
```

### 5. Final Output (Anggota 5)

```python
DLABenchmark:
    total_macs: int
    total_cycles: int
    throughput_gops: float          # Giga-OPS
    energy_total_uJ: float
    energy_efficiency_gops_w: float # GOPS/W
    pe_utilization: float
    memory_bound: bool
    layer_results: list             # per-layer breakdown
```

---

## Aturan Kolaborasi

1. **Jangan ubah interface tanpa diskusi.** Kalau butuh field baru,
   tambahkan di `common/interfaces.py` dan beritahu semua anggota.

2. **Gunakan LayerSpec untuk definisi layer.** Jangan hardcode dimensi
   layer di modul masing-masing.

3. **Parameter global ada di `config.py`.** Jangan duplikasi angka
   seperti clock frequency atau SRAM size di modul sendiri.

4. **Test modul sendiri dulu.** Jalankan `test_*.py` masing-masing
   sebelum integrasi.

5. **Dokumentasikan asumsi.** Kalau membuat simplifikasi (misal:
   "abaikan pipeline stall"), tulis di docstring fungsi.
