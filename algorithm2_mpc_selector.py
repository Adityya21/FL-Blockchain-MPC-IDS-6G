"""
Algorithm 2: Dynamic MPC Protocol Selection Engine
===================================================
Paper: FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight
       MPC-Secured Intrusion Detection System for 6G Networks (IEEE)
Authors: Param Udani, Aditya Yashovardhan, Aayush Maradia, et al.

Implements:
  - Multi-Party Computation (MPC) protocol candidate profiling (ASS, SSS, GC, PHE)
  - Dynamic weight derivation from real-time 6G network telemetry (Eq. 9)
  - Hard SLA constraint filtering
  - Multi-objective composite scoring function (Eq. 10)
  - Optimal protocol selection across 6G operating regimes (URLLC, eMBB, mMTC, Hostile)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class NetworkContext:
    """Real-time 6G Edge Network Context telemetry."""
    bandwidth_mbps: float  # BW in Mbps
    sla_latency_ms: float  # Latency SLA threshold in ms
    threat_level: float    # Threat index in [0.0, 10.0]
    battery_level: float   # Device energy/battery level in [0.0, 1.0]


@dataclass
class MPCProtocolProfile:
    """MPC Protocol Candidate benchmark profile (Table II in paper)."""
    name: str
    security_score: float   # S(P_i): [0.0, 10.0]
    comm_overhead_kb: float # C(P_i): Communication payload in KB
    latency_ms: float       # L(P_i): Computational + network latency in ms
    dropout_resilience: float # R(P_i): Resilience to client churn [0.0, 1.0]


# Table II: MPC Protocol Candidate Profiles
DEFAULT_CANDIDATES = [
    MPCProtocolProfile(name="ASS", security_score=6.5, comm_overhead_kb=12.5,  latency_ms=2.1,  dropout_resilience=0.40),
    MPCProtocolProfile(name="SSS", security_score=8.5, comm_overhead_kb=34.0,  latency_ms=14.8, dropout_resilience=0.95),
    MPCProtocolProfile(name="GC",  security_score=9.5, comm_overhead_kb=185.0, latency_ms=48.0, dropout_resilience=0.70),
    MPCProtocolProfile(name="PHE", security_score=8.0, comm_overhead_kb=95.0,  latency_ms=62.0, dropout_resilience=0.85),
]


class DynamicMPCSelector:
    """
    Algorithm 2: Dynamic MPC Protocol Selection Engine.
    Dynamically balances privacy/security, communication bandwidth, and latency/energy.
    """

    def __init__(self, candidates: Optional[List[MPCProtocolProfile]] = None, epsilon: float = 1e-6):
        self.candidates = candidates or DEFAULT_CANDIDATES
        self.epsilon = epsilon

        # Maximum values across candidates for normalization
        self.max_sec = 10.0
        self.max_comm = max(c.comm_overhead_kb for c in self.candidates)
        self.max_latency = max(c.latency_ms for c in self.candidates)

    def compute_weights(self, ctx: NetworkContext) -> Tuple[float, float, float]:
        """
        Derives normalized dynamic objective weights from network context using Eq. (9):
          w_sec  = threat / 10.0
          w_comm = max(0.2, 50.0 / (BW + eps))
          w_comp = max(0.2, (20.0 / (SLA + eps)) + (1.0 - batt))
        """
        w_sec = ctx.threat_level / 10.0
        w_comm = max(0.2, 50.0 / (ctx.bandwidth_mbps + self.epsilon))
        w_comp = max(0.2, (20.0 / (ctx.sla_latency_ms + self.epsilon)) + (1.0 - ctx.battery_level))

        # Normalization
        total_w = w_sec + w_comm + w_comp
        w_sec_norm = w_sec / total_w
        w_comm_norm = w_comm / total_w
        w_comp_norm = w_comp / total_w

        return (w_sec_norm, w_comm_norm, w_comp_norm)

    def compute_score(self, protocol: MPCProtocolProfile, weights: Tuple[float, float, float]) -> float:
        """
        Computes composite objective score for protocol candidate using Eq. (10):
          Score(P_i) = w_sec * S_norm(P_i) - w_comm * C_norm(P_i) - w_comp * L_norm(P_i)
        """
        w_sec, w_comm, w_comp = weights

        s_norm = protocol.security_score / self.max_sec
        c_norm = protocol.comm_overhead_kb / self.max_comm
        l_norm = protocol.latency_ms / self.max_latency

        score = (w_sec * s_norm) - (w_comm * c_norm) - (w_comp * l_norm)
        return score

    def select_protocol(self, ctx: NetworkContext) -> Tuple[MPCProtocolProfile, Dict[str, float], Tuple[float, float, float]]:
        """
        Executes Algorithm 2:
          1. Compute dynamic weights.
          2. Apply hard SLA filter: Latency(P_i) <= SLA.
          3. Evaluate composite score over eligible protocols.
          4. Select P* = argmax Score(P_i).
        """
        weights = self.compute_weights(ctx)

        # Hard SLA latency constraint filter
        eligible = [c for c in self.candidates if c.latency_ms <= ctx.sla_latency_ms]

        # Fallback if no candidate meets strict SLA: pick lowest latency candidate
        if not eligible:
            eligible = [min(self.candidates, key=lambda c: c.latency_ms)]

        scores = {}
        for candidate in self.candidates:
            if candidate in eligible:
                scores[candidate.name] = self.compute_score(candidate, weights)
            else:
                # Disqualified due to SLA violation
                scores[candidate.name] = -np.inf

        # Select protocol with highest score among eligible
        best_name = max(eligible, key=lambda c: scores[c.name]).name
        selected = next(c for c in self.candidates if c.name == best_name)

        return selected, scores, weights


def test_algorithm2_canonical_scenarios():
    """Validates Algorithm 2 on the 4 canonical 6G operating regimes from Table IX."""
    selector = DynamicMPCSelector()

    scenarios = {
        "URLLC (Ultra-Reliable Low-Latency)": NetworkContext(
            bandwidth_mbps=100.0,
            sla_latency_ms=5.0,
            threat_level=3.0,
            battery_level=0.8
        ),
        "eMBB (Enhanced Mobile Broadband)": NetworkContext(
            bandwidth_mbps=800.0,
            sla_latency_ms=50.0,
            threat_level=8.5,
            battery_level=0.9
        ),
        "mMTC (Massive Machine-Type Comm)": NetworkContext(
            bandwidth_mbps=5.0,
            sla_latency_ms=100.0,
            threat_level=2.0,
            battery_level=0.15
        ),
        "Hostile Defense (High Threat)": NetworkContext(
            bandwidth_mbps=200.0,
            sla_latency_ms=50.0,
            threat_level=10.0,
            battery_level=0.7
        )
    }

    print("=" * 80)
    print("ALGORITHM 2: DYNAMIC MPC PROTOCOL SELECTION VALIDATION (TABLE IX)")
    print("=" * 80)

    for name, ctx in scenarios.items():
        chosen, scores, weights = selector.select_protocol(ctx)
        print(f"\n[Scenario]: {name}")
        print(f"  Network Context: BW={ctx.bandwidth_mbps} Mbps, SLA={ctx.sla_latency_ms} ms, Threat={ctx.threat_level}/10, Battery={ctx.battery_level*100:.0f}%")
        print(f"  Derived Weights: w_sec={weights[0]:.2f}, w_comm={weights[1]:.2f}, w_comp={weights[2]:.2f}")
        print("  Protocol Evaluation:")
        for cand in selector.candidates:
            score_str = f"{scores[cand.name]:.4f}" if scores[cand.name] != -np.inf else "SLA_VIOLATED"
            status = " [CHOSEN]" if cand.name == chosen.name else ""
            print(f"    - {cand.name:4s} (Latency: {cand.latency_ms:4.1f}ms, Comm: {cand.comm_overhead_kb:5.1f}KB, Sec: {cand.security_score:3.1f}/10) -> Score: {score_str}{status}")
        print(f"  ==> Selected Protocol: {chosen.name} (Latency: {chosen.latency_ms} ms)")


if __name__ == "__main__":
    test_algorithm2_canonical_scenarios()
