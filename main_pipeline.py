"""
FBMP-IDS Unified End-to-End Pipeline
====================================
Paper: FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight
       MPC-Secured Intrusion Detection System for 6G Networks (IEEE)
Authors: Param Udani, Aditya Yashovardhan, Aayush Maradia, et al.

Executes all 6 architectural layers in sequence:
  Layer 1: CICIoT2023 Data Engine & Non-IID Dirichlet Partitioning
  Layer 2: Dynamic MPC Protocol Selection Engine (Algorithm 2)
  Layer 3: Cost-Sensitive Federated Learning Local Training
  Layer 4: INT8 Gradient Quantization (Payload Wire Compression)
  Layer 5: Additive Secret Sharing (ASS) MPC Zero-Knowledge Aggregation
  Layer 6: Consortium Blockchain Gradient Ledger & Tamper Audit
"""

import time
import numpy as np
from typing import Dict, List, Any

from algorithm2_mpc_selector import DynamicMPCSelector, NetworkContext
from mpc_engine import AdditiveSecretSharingMPC
from quantization import GradientQuantizer
from blockchain_ledger import BlockchainGradientLedger
from data_engine import CICIoT2023DataEngine, ATTACK_CATEGORIES
from federated_engine import CostSensitiveFLEngine


class FBMP_IDS_MasterPipeline:
    """
    Unified Master Pipeline integrating all architectural layers of FBMP-IDS.
    """

    def __init__(self, num_clients: int = 3, num_classes: int = 8, seed: int = 42):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.seed = seed

        # Core subsystem instances
        self.data_engine = CICIoT2023DataEngine(seed=seed)
        self.mpc_selector = DynamicMPCSelector()
        self.mpc_engine = AdditiveSecretSharingMPC(num_clients=num_clients, seed=seed)
        self.quantizer = GradientQuantizer()
        self.ledger = BlockchainGradientLedger()
        self.fl_engine = CostSensitiveFLEngine(num_classes=num_classes, num_features=39, seed=seed)

    def run(self, num_rounds: int = 4, test_tamper_attack: bool = True):
        """
        Executes end-to-end multi-round federated training matching Paper Section V.F & Table X.
        """
        print("=" * 90)
        print("FBMP-IDS UNIFIED END-TO-END PIPELINE EXECUTION (TABLE IV, V, VI, IX, X)")
        print("=" * 90)

        # 1. Dataset Preprocessing & Dirichlet Partitioning
        print("\n[Layer 1: Data Engine & Preprocessing]")
        X, y = self.data_engine.generate_benchmark_dataset(samples_per_class=1000)
        X_train, X_test, y_train, y_test = self.data_engine.preprocess_and_split(X, y, test_size=0.20)
        client_data = self.data_engine.partition_dirichlet_non_iid(
            X_train, y_train, num_clients=self.num_clients, alpha=0.5
        )
        print(f"  Dataset: {len(X)} flows | Features: 39 | Classes: 8")
        print(f"  Train: {len(X_train)} flows | Test: {len(X_test)} flows")
        print(f"  Partitioning: Non-IID Dirichlet (alpha=0.5) across K={self.num_clients} edge clients")

        round_history = []

        # Multi-Round Federated Pipeline
        for r in range(1, num_rounds + 1):
            print(f"\n{'='*35} ROUND {r}/{num_rounds} {'='*35}")

            # 2. Dynamic MPC Protocol Selection (Algorithm 2)
            # Simulate dynamic 6G telemetry conditions per round
            telemetry = [
                NetworkContext(bandwidth_mbps=100.0, sla_latency_ms=5.0,  threat_level=3.0, battery_level=0.85), # URLLC regime
                NetworkContext(bandwidth_mbps=400.0, sla_latency_ms=30.0, threat_level=7.5, battery_level=0.75), # Mixed/Threat regime
                NetworkContext(bandwidth_mbps=20.0,  sla_latency_ms=80.0, threat_level=2.5, battery_level=0.20), # Low-battery / mMTC regime
                NetworkContext(bandwidth_mbps=300.0, sla_latency_ms=40.0, threat_level=9.0, battery_level=0.70)  # Hostile regime
            ][(r - 1) % 4]

            chosen_proto, scores, weights = self.mpc_selector.select_protocol(telemetry)
            print(f"[Layer 2: Dynamic MPC Protocol Selection (Algorithm 2)]")
            print(f"  Telemetry: BW={telemetry.bandwidth_mbps}Mbps, SLA={telemetry.sla_latency_ms}ms, Threat={telemetry.threat_level}/10, Batt={telemetry.battery_level*100:.0f}%")
            print(f"  Optimal MPC Protocol Selected: {chosen_proto.name} (Latency: {chosen_proto.latency_ms}ms, Overhead: {chosen_proto.comm_overhead_kb}KB)")

            # 3. Cost-Sensitive Local Client Training
            print(f"[Layer 3: Edge Tier Cost-Sensitive Local Training]")
            client_coefs = []
            client_intercepts = []
            quantized_payloads = []
            client_metadata = []

            for k in range(self.num_clients):
                X_k, y_k = client_data[k]
                coef_k, inter_k = self.fl_engine.train_client_local(
                    X_k, y_k, cost_sensitive=True, epochs=5, lr=0.05
                )
                client_coefs.append(coef_k)
                client_intercepts.append(inter_k)

                # 4. INT8 Gradient Quantization
                full_weights = np.hstack([coef_k, inter_k[:, None]])
                q_payload = self.quantizer.quantize(full_weights)
                quantized_payloads.append(q_payload)

                client_metadata.append({
                    "client_id": k + 1,
                    "sample_count": len(X_k),
                    "quantized_bytes": q_payload.payload_bytes,
                    "scale": float(q_payload.scale),
                    "zero_point": int(q_payload.zero_point)
                })

            raw_bytes_per_client = full_weights.size * 4
            q_bytes_per_client = quantized_payloads[0].payload_bytes
            savings_pct = (1.0 - (q_bytes_per_client / raw_bytes_per_client)) * 100.0
            print(f"[Layer 4: INT8 Gradient Quantization]")
            print(f"  Uncompressed Payload: {raw_bytes_per_client} B -> Quantized: {q_bytes_per_client} B (Savings: {savings_pct:.2f}%)")

            # 5. Additive Secret Sharing MPC Secure Aggregation
            print(f"[Layer 5: Additive Secret Sharing MPC Aggregation]")
            # Dequantize weights for aggregation
            dequantized_weights = [self.quantizer.dequantize(p) for p in quantized_payloads]

            # Split parameters
            flat_weights = [w.flatten() for w in dequantized_weights]
            reconstructed_flat, abs_error, rel_error = self.mpc_engine.secure_aggregate(flat_weights)
            reconstructed_matrix = reconstructed_flat.reshape(full_weights.shape)

            recon_coef = reconstructed_matrix[:, :39]
            recon_inter = reconstructed_matrix[:, 39]

            # Update global FL model
            self.fl_engine.update_global_model([recon_coef], [recon_inter])
            print(f"  MPC Reconstruction Max Error (Delta): {abs_error:.2e} (Machine Precision)")

            # 6. Consortium Blockchain Gradient Ledger Mining
            print(f"[Layer 6: Consortium Blockchain Ledger Mining]")
            mined_block = self.ledger.commit_fl_round(
                round_num=r,
                model_weights=reconstructed_flat,
                client_metadata=client_metadata
            )
            print(f"  Mined Block #{mined_block.index}: Merkle Root = {mined_block.merkle_root[:16]}..., Hash = {mined_block.block_hash[:16]}...")

            # Evaluate Global Model
            eval_metrics = self.fl_engine.evaluate(X_test, y_test)
            print(f"[Round {r} Evaluation Results]")
            print(f"  Accuracy:  {eval_metrics['accuracy'] * 100:.2f}% (Paper Target: ~95-96%)")
            print(f"  Precision: {eval_metrics['precision'] * 100:.2f}%")
            print(f"  Recall:    {eval_metrics['recall'] * 100:.2f}% (Paper Target: ~98%)")
            print(f"  Macro F1:  {eval_metrics['f1_macro'] * 100:.2f}%")

            round_history.append({
                "round": r,
                "protocol": chosen_proto.name,
                "accuracy": eval_metrics['accuracy'],
                "precision": eval_metrics['precision'],
                "recall": eval_metrics['recall'],
                "f1": eval_metrics['f1_macro'],
                "mpc_error": abs_error,
                "savings_pct": savings_pct,
                "block_index": mined_block.index
            })

        # Summary Table matching Table X & Table IV
        print("\n" + "=" * 90)
        print("SUMMARY: UNIFIED PIPELINE ROUND-BY-ROUND METRICS (TABLE X & IV)")
        print("=" * 90)
        print(f"{'Round':6s} | {'MPC Protocol':13s} | {'Accuracy':9s} | {'Precision':10s} | {'Recall':8s} | {'F1':8s} | {'MPC Error':11s} | {'INT8 Savings':12s}")
        print("-" * 90)
        for rh in round_history:
            print(f"{rh['round']:6d} | {rh['protocol']:13s} | {rh['accuracy']*100:8.2f}% | {rh['precision']*100:9.2f}% | {rh['recall']*100:7.2f}% | {rh['f1']*100:7.2f}% | {rh['mpc_error']:.2e} | {rh['savings_pct']:11.2f}%")

        # Blockchain Ledger Complete Audit
        print("\n" + "=" * 90)
        print("BLOCKCHAIN INTEGRITY & AUDIT (TABLE V)")
        print("=" * 90)
        is_valid, audit_logs = self.ledger.audit_chain()
        print(f"Consortium Ledger Audit Status: {'[PASSED] 100% VALID' if is_valid else 'FAILED'}")
        for log in audit_logs:
            print(f"  {log}")

        # Tamper Attack Simulation
        if test_tamper_attack:
            print("\n[Simulating Model Poisoning / Block Tamper Attack on Block #2...]")
            attack_detected, msg = self.ledger.simulate_tamper_attack(target_block_idx=2, tampered_weight_val=9999.9999)
            print(f"Tamper Audit Verdict: {msg}")
            print(f"Security Result: {'[PASSED] Tampered update REJECTED -> Model poisoning prevented' if attack_detected else 'VULNERABLE'}")


def main():
    pipeline = FBMP_IDS_MasterPipeline(num_clients=3, num_classes=8, seed=42)
    pipeline.run(num_rounds=4, test_tamper_attack=True)


if __name__ == "__main__":
    main()
