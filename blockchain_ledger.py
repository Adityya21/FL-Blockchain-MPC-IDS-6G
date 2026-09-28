"""
Consortium Blockchain Gradient Ledger with Merkle Tree & Tamper Proofing
========================================================================
Paper: FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight
       MPC-Secured Intrusion Detection System for 6G Networks (IEEE)
Authors: Param Udani, Aditya Yashovardhan, Aayush Maradia, et al.

Implements:
  - Cryptographic block structures with SHA-256 hashing
  - Binary Merkle Tree for verifiable gradient transaction sets
  - Consortium HMAC-SHA256 signatures for node authorization
  - Cascading tamper-evident chain verification (Table V)
  - Model poisoning / gradient tampering attack detection and rejection
"""

import hashlib
import hmac
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class MerkleTree:
    """
    Binary Merkle Tree for Cryptographic Verification of Gradient Transactions.
    """

    def __init__(self, transactions: List[str]):
        self.transactions = transactions
        self.root = self.build_tree(transactions)

    @staticmethod
    def sha256(data: str) -> str:
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def build_tree(self, leaves: List[str]) -> str:
        if not leaves:
            return self.sha256("")
        
        current_layer = [self.sha256(tx) for tx in leaves]

        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                left = current_layer[i]
                # If odd number of nodes, duplicate the last one
                right = current_layer[i + 1] if i + 1 < len(current_layer) else left
                combined = self.sha256(left + right)
                next_layer.append(combined)
            current_layer = next_layer

        return current_layer[0]

    def get_root(self) -> str:
        return self.root


class Block:
    """
    Consortium Blockchain Block Header and Body.
    """

    def __init__(
        self,
        index: int,
        previous_hash: str,
        round_num: int,
        transactions: List[Dict[str, Any]],
        secret_key: bytes,
        timestamp: Optional[float] = None
    ):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.round_num = round_num
        self.transactions = transactions
        self.previous_hash = previous_hash

        # Compute Merkle Root
        tx_strings = [json.dumps(tx, sort_keys=True) for tx in transactions]
        self.merkle_tree = MerkleTree(tx_strings)
        self.merkle_root = self.merkle_tree.get_root()

        # Compute Block Hash
        self.block_hash = self.compute_hash()

        # Compute Consortium HMAC Signature
        self.hmac_signature = self.compute_signature(secret_key)

    def compute_hash(self) -> str:
        """SHA-256 over header fields."""
        header = f"{self.index}:{self.previous_hash}:{self.timestamp}:{self.round_num}:{self.merkle_root}"
        return hashlib.sha256(header.encode('utf-8')).hexdigest()

    def compute_signature(self, secret_key: bytes) -> str:
        """HMAC-SHA256 consortium authorization signature."""
        msg = f"{self.block_hash}:{self.merkle_root}".encode('utf-8')
        return hmac.new(secret_key, msg, hashlib.sha256).hexdigest()

    def verify_block(self, secret_key: bytes) -> Tuple[bool, str]:
        """Verifies block hash integrity, Merkle root, and HMAC signature."""
        # 1. Verify block hash
        expected_hash = self.compute_hash()
        if self.block_hash != expected_hash:
            return False, f"Block #{self.index} hash mismatch (Expected: {expected_hash}, Actual: {self.block_hash})"

        # 2. Recompute Merkle root
        tx_strings = [json.dumps(tx, sort_keys=True) for tx in self.transactions]
        computed_root = MerkleTree(tx_strings).get_root()
        if self.merkle_root != computed_root:
            return False, f"Block #{self.index} Merkle root corrupted!"

        # 3. Verify HMAC signature
        expected_sig = self.compute_signature(secret_key)
        if not hmac.compare_digest(self.hmac_signature, expected_sig):
            return False, f"Block #{self.index} HMAC-SHA256 signature invalid!"

        return True, "VALID"


class BlockchainGradientLedger:
    """
    Consortium Blockchain Gradient Ledger for FBMP-IDS.
    Guarantees tamper-proof model integrity and prevents gradient poisoning.
    """

    def __init__(self, consortium_secret: bytes = b"fbmp-ids-consortium-secret-key-6g"):
        self.consortium_secret = consortium_secret
        self.chain: List[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Mines genesis block (Block #0)."""
        genesis_tx = [{
            "type": "GENESIS",
            "message": "FBMP-IDS Consortium Ledger Initialized",
            "timestamp": 1700000000.0
        }]
        genesis_block = Block(
            index=0,
            previous_hash="0" * 64,
            round_num=0,
            transactions=genesis_tx,
            secret_key=self.consortium_secret,
            timestamp=1700000000.0
        )
        self.chain.append(genesis_block)

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def commit_fl_round(self, round_num: int, model_weights: np.ndarray, client_metadata: List[Dict[str, Any]]) -> Block:
        """
        Creates, signs, and commits a new block with the aggregated FL model and client proofs.
        """
        weight_hash = hashlib.sha256(model_weights.tobytes()).hexdigest()

        transactions = [
            {
                "type": "AGGREGATED_MODEL_UPDATE",
                "round": round_num,
                "model_weight_sha256": weight_hash,
                "weights_sample": model_weights.flatten()[:5].tolist(),
                "num_parameters": model_weights.size,
                "client_metadata": client_metadata
            }
        ]

        prev_block = self.get_latest_block()
        new_block = Block(
            index=len(self.chain),
            previous_hash=prev_block.block_hash,
            round_num=round_num,
            transactions=transactions,
            secret_key=self.consortium_secret
        )

        self.chain.append(new_block)
        return new_block

    def audit_chain(self) -> Tuple[bool, List[str]]:
        """
        Audits complete blockchain ledger:
          - Verifies hash linking (previous_hash == chain[i-1].block_hash)
          - Verifies internal block hash and Merkle root
          - Verifies consortium HMAC-SHA256 signature
        """
        logs = []
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # 1. Chain continuity
            if current.previous_hash != prev.block_hash:
                err = f"Chain Discontinuity at Block #{current.index}: prev_hash does not match Block #{prev.index} hash!"
                logs.append(err)
                return False, logs

            # 2. Block internal integrity
            is_valid, msg = current.verify_block(self.consortium_secret)
            if not is_valid:
                logs.append(msg)
                return False, logs

            logs.append(f"Block #{current.index} [Round {current.round_num}] -> AUDIT PASSED")

        return True, logs

    def simulate_tamper_attack(self, target_block_idx: int = 1, tampered_weight_val: float = 9999.9999) -> Tuple[bool, str]:
        """
        Simulates adversarial model poisoning by tampering with model parameters in a committed block.
        Paper Section V.B: Weight altered to 9999.9999 in Block #2 -> Promptly REJECTED.
        """
        if target_block_idx >= len(self.chain):
            return False, "Target block index out of range."

        target_block = self.chain[target_block_idx]

        # Adversary mutates transaction weights
        target_block.transactions[0]["weights_sample"][0] = tampered_weight_val
        target_block.transactions[0]["model_weight_sha256"] = "tampered_hash_9999"

        # Run blockchain consensus audit
        is_valid, logs = self.audit_chain()

        if not is_valid:
            return True, f"TAMPER ATTACK DETECTED AND BLOCKED: {logs[-1]}"
        else:
            return False, "Tamper attack undetected (Consensus failure!)"


def test_blockchain_ledger():
    """Unit test validating Table V blockchain audit results."""
    print("=" * 80)
    print("BLOCKCHAIN LEDGER AUDIT & TAMPER PROOFING VALIDATION (TABLE V)")
    print("=" * 80)

    ledger = BlockchainGradientLedger()

    # Commit 3 FL rounds
    np.random.seed(42)
    for r in range(1, 4):
        fake_weights = np.random.normal(loc=0.0, scale=1.0, size=(40,))
        client_meta = [{"client_id": k, "status": "SHARES_VERIFIED"} for k in range(3)]
        blk = ledger.commit_fl_round(round_num=r, model_weights=fake_weights, client_metadata=client_meta)
        print(f"Mined Block #{blk.index} [Round {r}]: Merkle Root = {blk.merkle_root[:16]}..., Hash = {blk.block_hash[:16]}...")

    print(f"\nTotal Blocks: {len(ledger.chain)} (Genesis + 3 Rounds)")

    # Audit valid chain
    is_valid, logs = ledger.audit_chain()
    print(f"Initial Ledger Audit: {'[PASSED] 100% VALID' if is_valid else 'FAILED'}")
    for log in logs:
        print(f"  {log}")

    # Simulate adversarial poisoning attack (Paper Section V.B)
    print("\n[Simulating Adversarial Model Poisoning Attack on Block #2...]")
    tamper_detected, report = ledger.simulate_tamper_attack(target_block_idx=2, tampered_weight_val=9999.9999)
    print(f"Attack Result: {report}")
    print(f"Consensus Verdict: {'[PASSED] TAMPERED UPDATE REJECTED (Poisoning Prevented)' if tamper_detected else 'VULNERABLE'}")


if __name__ == "__main__":
    test_blockchain_ledger()
