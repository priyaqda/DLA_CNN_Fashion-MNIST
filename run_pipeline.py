"""
run_pipeline.py — Satu perintah: training → quantize → export → verify.

Jalankan:
    cd dla_level1
    python run_pipeline.py

Pipeline:
    Step 1: Training LeNet-5 pada Fashion-MNIST       → weights/lenet5_fp32.npy
    Step 2: Quantize FP32 → INT8 + export HEX         → weights/lenet5_int8.hex
    Step 3: Verify inference FP32 vs INT8              → accuracy report
    Step 4: Run DLA benchmark                          → performance report

Setelah selesai, file weights/lenet5_int8.hex siap dipakai di Verilog:
    $readmemh("lenet5_int8.hex", weight_rom);
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step_banner(step_num, title):
    print(f"\n{'▓'*60}")
    print(f"  STEP {step_num}/4 — {title}")
    print(f"{'▓'*60}\n")


def main():
    start = time.time()

    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  DLA CNN Accelerator — Full Pipeline                     ║")
    print("║  Fashion-MNIST → Training → INT8 → HEX → Verilog-ready  ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # ── STEP 1: Training ─────────────────────────────────────
    step_banner(1, "TRAINING LeNet-5")

    from anggota4_quantization.train_lenet import train
    model, accuracy = train(epochs=10, batch_size=64, lr=1e-3)

    if not os.path.exists("weights/lenet5_fp32.npy"):
        print("❌ Training gagal — weights tidak tersimpan")
        sys.exit(1)

    # ── STEP 2: Quantize + Export ────────────────────────────
    step_banner(2, "QUANTIZE & EXPORT")

    from anggota4_quantization.export_weights import main as export_main
    export_main()

    if not os.path.exists("weights/lenet5_int8.hex"):
        print("❌ Export gagal — HEX file tidak tersimpan")
        sys.exit(1)

    # ── STEP 3: Verify Inference ─────────────────────────────
    step_banner(3, "VERIFY INFERENCE (FP32 vs INT8)")

    from anggota1_pe_array.inference import main as inference_main
    inference_main()

    # ── STEP 4: DLA Benchmark ────────────────────────────────
    step_banner(4, "DLA BENCHMARK")

    from anggota5_integration.run_demo import main as benchmark_main
    benchmark_main()

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start

    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  PIPELINE COMPLETE                                       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\n  Total time     : {elapsed:.1f} seconds")
    print(f"  Training acc   : {accuracy:.2f}%")
    print(f"\n  Output files:")
    print(f"    weights/lenet5_fp32.npy     — FP32 weights (referensi)")
    print(f"    weights/lenet5_int8.npy     — INT8 weights (Python sim)")
    print(f"    weights/lenet5_int8.hex     — INT8 HEX (Verilog ROM)")
    print(f"    weights/lenet5_scales.hex   — Scale factors (dequant)")
    print(f"    weights/weight_summary.txt  — Ringkasan lengkap")
    print(f"    weights/accuracy.txt        — Training log")
    print(f"\n  Next step (Level 2 — Verilog):")
    print(f"    reg [7:0] w_rom [0:N];")
    print(f"    initial $readmemh(\"lenet5_int8.hex\", w_rom);")
    print(f"\n")


if __name__ == "__main__":
    main()
