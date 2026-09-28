# FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight MPC-Secured Intrusion Detection System for 6G Networks

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE--Formatted--v2-red.svg)](FBMP_IDS_IEEE_Formatted_v2.pdf)
[![Reproducibility](https://img.shields.io/badge/Results-100%25--Reproducible-brightgreen.svg)](reproduce_all_tables.py)

---

## 📖 Overview

In the era of 6G networks and massive Internet of Things (IoT) ecosystems, Intrusion Detection Systems (IDS) face stringent challenges: ultra-low latency guarantees (URLLC), heterogeneous non-IID edge telemetry, extreme class imbalance, communication bandwidth bottlenecks, and vulnerability to model poisoning attacks.

**FBMP-IDS** provides a fully implemented and empirically validated framework uniting:
1. **Dynamic MPC Protocol Selection Engine (Algorithm 2)**: Dynamically selects optimal Multi-Party Computation schemes (`ASS`, `SSS`, `GC`, `PHE`) across 6G operating regimes (URLLC, eMBB, mMTC, Hostile Defense) based on real-time bandwidth, SLA deadlines, threat metrics, and battery state.
2. **Additive Secret Sharing (ASS) MPC Engine**: Zero-knowledge gradient aggregation across $K$ edge devices with exact numerical reconstruction ($\Delta < 10^{-14}$).
3. **Cost-Sensitive Federated Learning**: Remediates natural 6G traffic class imbalance using inverse-frequency sample weighting, boosting minority attack detection (**+11.75% BruteForce**, **+11.50% Spoofing**).
4. **INT8 Gradient Quantization**: Uniform affine FP32-to-INT8 compression yielding **73.75% wire payload reduction** (1,280 B $\to$ 336 B) at **38.89 dB SNR**.
5. **Consortium Blockchain Gradient Ledger**: Tamper-proof gradient anchoring using binary Merkle trees, SHA-256 block chaining, and HMAC-SHA256 signatures, preventing model poisoning.
6. **CICIoT2023 6G Benchmark Engine**: Memory-safe processing and Non-IID Dirichlet distribution ($\alpha = 0.5$) across 8 attack categories and 39 network flow features.

---

## 🏛️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                               1. EDGE-IOT TIER                                    |
|  K Edge Devices train local SGD classifiers with Cost-Sensitive Sample Weighting  |
|               theta_k^(t+1) = theta_k^(t) - eta * sum(w_c * grad_loss)            |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                         2. GRADIENT QUANTIZATION TIER                             |
|          FP32 -> INT8 Uniform Affine Quantization (73.75% Wire Compression)        |
|               q = clip(round(w / scale) + zero_point, -128, 127)                   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        3. DYNAMIC MPC & ZERO-KNOWLEDGE TIER                       |
|   Algorithm 2 evaluates SLA, Bandwidth, Threat, Battery -> Selects Protocol      |
|   Additive Secret Sharing (ASS): W_k split into K blinded random noise shares    |
|   Aggregated Share: A_j = sum(S_{k,j})  ==>  Exact Reconstruction (Delta < 10^-14)|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                       4. CONSORTIUM BLOCKCHAIN LEDGER TIER                        |
|  Merkle Root Verification + HMAC-SHA256 Signatures + Cryptographic Hash Chaining  |
|  Cascading Tamper Detection: Instantly detects and rejects Poisoning Attacks     |
+-----------------------------------------------------------------------------------+
```

---

## 🧮 Mathematical Formulations

### 1. Cost-Sensitive Local Training (Eq. 1 & 2)
$$\theta_k^{(t+1)} = \theta_k^{(t)} - \eta \sum_{i} w_{y_i} \nabla \ell(f_\theta(x_i), y_i)$$
$$w_c = \frac{N}{C \times N_c}$$

### 2. Additive Secret Sharing MPC (Eq. 3, 4, 5)
$$W_k = \sum_{j=1}^K S_{k,j} = S_{k,1} + S_{k,2} + \dots + S_{k,K}$$
$$S_{k,K} = W_k - \sum_{j=1}^{K-1} S_{k,j}, \quad S_{k,j} \sim \mathcal{N}(0, \sigma^2 I)$$
$$\bar{W} = \frac{1}{K} \sum_{j=1}^K A_j = \frac{1}{K} \sum_{k=1}^K W_k \quad (\Delta < 10^{-14})$$

### 3. INT8 Gradient Quantization (Eq. 6, 7, 8)
$$\mathrm{scale} = \frac{w_{\max} - w_{\min}}{255.0}, \quad z_{\mathrm{point}} = \mathrm{round}\left(-\frac{w_{\min}}{\mathrm{scale}}\right) - 128$$
$$q = \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{\mathrm{scale}}\right) + z_{\mathrm{point}}, -128, 127\right)$$
$$\hat{w} = (q - z_{\mathrm{point}}) \times \mathrm{scale}$$
$$\mathrm{SNR} = 10 \log_{10} \left[ \frac{\mathrm{Var}(W)}{\mathbb{E}[(W - \hat{W})^2]} \right] \text{ dB}$$

### 4. Algorithm 2 Dynamic Context Scoring (Eq. 9 & 10)
$$w_{\text{sec}} = \frac{\text{threat}}{10.0}, \quad w_{\text{comm}} = \max\left(0.2, \frac{50}{BW + \varepsilon}\right), \quad w_{\text{comp}} = \max\left(0.2, \frac{20}{SLA + \varepsilon} + (1 - \text{batt})\right)$$
$$\text{Score}(P_i) = \tilde{w}_{\text{sec}} \cdot \bar{S}(P_i) - \tilde{w}_{\text{comm}} \cdot \bar{C}(P_i) - \tilde{w}_{\text{comp}} \cdot \bar{L}(P_i)$$
$$\text{subject to: } \text{Latency}(P_i) \le SLA$$

---

## 📁 Repository Structure

```
FBMP-IDS-6G/
│
├── algorithm2_mpc_selector.py   # Algorithm 2: Dynamic MPC Protocol Selection Engine
├── mpc_engine.py                # Additive Secret Sharing (ASS) MPC Engine
├── quantization.py              # Uniform Affine INT8 Gradient Quantization
├── blockchain_ledger.py         # Consortium Blockchain Ledger & Tamper Proofing
├── federated_engine.py          # Cost-Sensitive Federated Learning Engine
├── data_engine.py               # CICIoT2023 Non-IID Dirichlet Partitioning Engine
├── binary_fl_benchmark.py       # Binary FL+MPC Benchmark (Table IV)
├── main_pipeline.py             # Unified End-to-End Master Pipeline
├── reproduce_all_tables.py      # Master reproduction runner for all paper tables
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.9+
- NumPy, Scikit-learn, Pandas

### Installation
```bash
git clone https://github.com/<your-username>/FBMP-IDS-6G.git
cd FBMP-IDS-6G
pip install -r requirements.txt
```

### Reproduce All Results
Execute the master reproduction script to run all 6 layers and verify all tables:
```bash
python reproduce_all_tables.py
```

### Run Individual Modules

1. **Algorithm 2 Dynamic Protocol Selector**:
   ```bash
   python algorithm2_mpc_selector.py
   ```
2. **Additive MPC Engine**:
   ```bash
   python mpc_engine.py
   ```
3. **INT8 Gradient Quantization**:
   ```bash
   python quantization.py
   ```
4. **Blockchain Ledger & Tamper Attack Audit**:
   ```bash
   python blockchain_ledger.py
   ```
5. **Class Imbalance Remediation**:
   ```bash
   python federated_engine.py
   ```
6. **Unified End-to-End Pipeline**:
   ```bash
   python main_pipeline.py
   ```

---

## 📊 Experimental Results Verification

### Table II & IX: Algorithm 2 Dynamic MPC Protocol Selection
| Scenario | Constraint | Selected Protocol | Latency | Overhead | Security |
|:---|:---|:---:|:---:|:---:|:---:|
| **URLLC (5ms SLA)** | Strict Latency | **ASS** | **2.1 ms** | 12.5 KB | 6.5 / 10 |
| **eMBB (800 Mbps)** | High Threat | **SSS** | 14.8 ms | 34.0 KB | **8.5 / 10** |
| **mMTC (5 Mbps)** | BW & Battery | **ASS** | **2.1 ms** | **12.5 KB** | 6.5 / 10 |
| **Hostile (10/10)** | Maximum Defense | **SSS** | 14.8 ms | 34.0 KB | **8.5 / 10** |

### Table IV: FL + MPC Aggregation Performance (Binary CICIoT2023)
| Round | Accuracy | Precision | Recall | F1 Score | MPC Reconstruction Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 95.33% | 93.59% | 97.33% | 95.42% | $7.1 \times 10^{-15}$ |
| 2 | 94.00% | 92.31% | 96.00% | 94.12% | $1.4 \times 10^{-14}$ |
| 3 | 95.00% | 92.45% | 98.00% | 95.15% | $3.6 \times 10^{-15}$ |
| 4 | **96.33%** | **94.84%** | **98.00%** | **96.39%** | **$7.1 \times 10^{-15}$** |

### Table V: Blockchain Ledger Audit & Tamper Proofing
| Metric | Benchmark Result |
|:---|:---:|
| Total Blocks | 4 (Genesis + 3 FL Rounds) |
| Hash Algorithm | SHA-256 |
| Signature Scheme | HMAC-SHA256 |
| Merkle Verification | **[PASSED] All Blocks Valid** |
| Chain Continuity | **[PASSED] 100% Intact** |
| Tamper Attack (Weight = 9999.9999) | **REJECTED (Poisoning Prevented)** |

### Table VI & VII: Gradient Compression & INT8 Quantization
| Precision | Payload (Bytes) | Compression Ratio | Bandwidth Savings | SNR |
|:---|:---:|:---:|:---:|:---:|
| **FP32 (Baseline)** | 1,280 B | 1.00x | 0.00% | $\infty$ |
| **FP16 (Half)** | 640 B | 2.00x | 50.00% | -- |
| **INT8 (Quantized)** | **336 B** | **3.81x** | **73.75%** | **38.89 dB** |

### Table VIII: Class Imbalance Remediation (Detection Rate / TPR)
| Attack Category | Training Flows | Standard TPR | Cost-Sensitive TPR | Detection Gain |
|:---|:---:|:---:|:---:|:---:|
| **BruteForce** | 160 | 9.25% | **21.00%** | **+11.75%** |
| **Spoofing** | 240 | 65.75% | **77.25%** | **+11.50%** |
| **WebBased** | 320 | 20.50% | **23.50%** | **+3.00%** |
| **Macro F1** | 6,880 | 60.26% | **61.71%** | **+1.46%** |

---

## 📝 Citation

```bibtex
@article{udani2026fbmp,
  title={FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight MPC-Secured Intrusion Detection System for 6G Networks -- Implementation, Validation, and Enhanced Results},
  author={Udani, Param and Yashovardhan, Aditya and Maradia, Aayush and Gurjar, Daksh and Mehta, Rishabh and Desai, Kunj},
  journal={IEEE Transactions / Access},
  year={2026}
}
```
