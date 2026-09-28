"""
FBMP-IDS Master Reproduction Script
===================================
Paper: FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight
       MPC-Secured Intrusion Detection System for 6G Networks (IEEE)
Authors: Param Udani, Aditya Yashovardhan, Aayush Maradia, et al.

Executes and verifies all experimental evaluations and generates all
results tables (Tables II, IV, V, VI, VII, VIII, IX, X) reported in the paper.
"""

import sys
import numpy as np

from algorithm2_mpc_selector import test_algorithm2_canonical_scenarios
from mpc_engine import test_mpc_engine
from quantization import test_quantization_engine
from blockchain_ledger import test_blockchain_ledger
from federated_engine import benchmark_class_imbalance
from binary_fl_benchmark import run_binary_fl_mpc_benchmark
from main_pipeline import main as run_unified_pipeline


def print_banner(title: str):
    print("\n" + "#" * 90)
    print(f"# {title.center(86)} #")
    print("#" * 90 + "\n")


def main():
    print("=" * 90)
    print("REPRODUCING ALL RESULTS FOR FBMP-IDS (IEEE FORMATTED V2)")
    print("=" * 90)

    # 1. Algorithm 2: Dynamic MPC Protocol Selection
    print_banner("1. TABLE II & IX: DYNAMIC MPC PROTOCOL SELECTION (ALGORITHM 2)")
    test_algorithm2_canonical_scenarios()

    # 2. Additive Secret Sharing MPC Engine
    print_banner("2. MODULE 1: ADDITIVE SECRET SHARING (ASS) MPC PRECISION")
    test_mpc_engine()

    # 3. Binary FL + MPC Convergence
    print_banner("3. TABLE IV: FL + MPC AGGREGATION PERFORMANCE (BINARY, CICIoT2023)")
    run_binary_fl_mpc_benchmark()

    # 4. INT8 Gradient Quantization & Compression
    print_banner("4. TABLE VI & VII: INT8 GRADIENT QUANTIZATION & COMPRESSION")
    test_quantization_engine()

    # 5. Class Imbalance Remediation
    print_banner("5. TABLE VIII: CLASS IMBALANCE REMEDIATION (PER-CLASS TPR)")
    benchmark_class_imbalance()

    # 6. Consortium Blockchain Ledger & Tamper Detection
    print_banner("6. TABLE V: BLOCKCHAIN LEDGER AUDIT & TAMPER ATTACK DETECTION")
    test_blockchain_ledger()

    # 7. Unified Master Pipeline
    print_banner("7. TABLE X: UNIFIED END-TO-END 6-LAYER PIPELINE")
    run_unified_pipeline()

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTAL VALIDATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 90)


if __name__ == "__main__":
    main()
