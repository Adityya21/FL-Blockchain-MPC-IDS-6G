"""
Cost-Sensitive Federated Learning (FL) Engine
=============================================
Paper: FBMP-IDS: A Federated Learning-Based Blockchain-Powered Lightweight
       MPC-Secured Intrusion Detection System for 6G Networks (IEEE)
Authors: Param Udani, Aditya Yashovardhan, Aayush Maradia, et al.

Implements:
  - Cost-sensitive sample weighting via inverse class frequency (Eq. 1 & 2)
  - Local SGDClassifier training on edge nodes (logistic loss, E=5 epochs)
  - Federal aggregation of model weights and intercepts
  - Class-imbalance remediation benchmark matching Table VIII
  - Multi-round evaluation (Accuracy, Precision, Recall, F1, Per-class TPR)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CostSensitiveFLEngine:
    """
    Cost-Sensitive Federated Learning Engine for 6G Intrusion Detection.
    Remediates extreme class imbalance across heterogeneous edge nodes.
    """

    def __init__(self, num_classes: int = 8, num_features: int = 39, seed: int = 42):
        self.num_classes = num_classes
        self.num_features = num_features
        self.seed = seed

        # Global model parameter arrays: coef_ (num_classes, num_features) and intercept_ (num_classes,)
        self.global_coef = np.zeros((num_classes, num_features), dtype=np.float64)
        self.global_intercept = np.zeros((num_classes,), dtype=np.float64)

    @staticmethod
    def compute_class_weights(y: np.ndarray, num_classes: int = 8) -> np.ndarray:
        """
        Computes inverse class frequency weights (Eq. 2):
          w_c = N / (C * N_c)
        """
        N = len(y)
        counts = np.bincount(y, minlength=num_classes)
        weights = np.zeros(num_classes, dtype=np.float64)

        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = N / (num_classes * counts[c])
            else:
                weights[c] = 1.0

        return weights

    def train_client_local(
        self,
        X_k: np.ndarray,
        y_k: np.ndarray,
        cost_sensitive: bool = True,
        epochs: int = 5,
        lr: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains local SGDClassifier on Client k for E epochs with cost-sensitive weighting (Eq. 1).
        """
        clf = SGDClassifier(
            loss='log_loss',
            learning_rate='constant',
            eta0=lr,
            max_iter=epochs,
            tol=None,
            random_state=self.seed,
            warm_start=True
        )

        # Initialize with current global weights if available
        all_classes = np.arange(self.num_classes)
        clf.classes_ = all_classes
        clf.coef_ = self.global_coef.copy()
        clf.intercept_ = self.global_intercept.copy()

        # Compute sample weights if cost-sensitive
        if cost_sensitive:
            class_weights = self.compute_class_weights(y_k, self.num_classes)
            sample_weight = class_weights[y_k]
        else:
            sample_weight = None

        # Execute local training epochs
        clf.fit(X_k, y_k, sample_weight=sample_weight)

        return clf.coef_.copy(), clf.intercept_.copy()

    def update_global_model(self, coef_list: List[np.ndarray], intercept_list: List[np.ndarray]):
        """
        Updates global model using aggregated parameters (from FedAvg or reconstructed MPC).
        """
        self.global_coef = np.mean(coef_list, axis=0)
        self.global_intercept = np.mean(intercept_list, axis=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates current global model on test set:
          - Overall Accuracy, Precision, Recall, Macro F1
          - Per-class True Positive Rate (TPR / Detection Rate)
        """
        # Linear decision: argmax(X @ W^T + b)
        scores = np.dot(X_test, self.global_coef.T) + self.global_intercept
        y_pred = np.argmax(scores, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Per-class TPR
        per_class_tpr = {}
        for c in range(self.num_classes):
            mask = (y_test == c)
            if np.sum(mask) > 0:
                tpr = float(np.sum(y_pred[mask] == c) / np.sum(mask))
            else:
                tpr = 0.0
            per_class_tpr[c] = tpr

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1,
            "per_class_tpr": per_class_tpr,
            "y_pred": y_pred
        }


def benchmark_class_imbalance():
    """
    Direct benchmark comparing Standard Unweighted Training vs Cost-Sensitive Balanced Learning.
    Matches Table VIII results in paper.
    """
    from data_engine import CICIoT2023DataEngine, ATTACK_CATEGORIES

    print("=" * 80)
    print("CLASS IMBALANCE REMEDIATION: PER-CLASS TPR COMPARISON (TABLE VIII)")
    print("=" * 80)

    engine = CICIoT2023DataEngine(seed=42)
    # Generate imbalanced train set (6,880 flows) and balanced test set
    X_train_raw, y_train = engine.generate_benchmark_dataset(imbalanced=True)
    X_test_raw, y_test = engine.generate_benchmark_dataset(samples_per_class=400, imbalanced=False)

    # Scale
    X_train = engine.scaler.fit_transform(X_train_raw)
    X_test = engine.scaler.transform(X_test_raw)

    # 1. Standard Unweighted Training
    fl_standard = CostSensitiveFLEngine(num_classes=8, num_features=39, seed=42)
    coef_std, inter_std = fl_standard.train_client_local(X_train, y_train, cost_sensitive=False, epochs=10)
    fl_standard.update_global_model([coef_std], [inter_std])
    res_std = fl_standard.evaluate(X_test, y_test)

    # 2. Cost-Sensitive Balanced Learning
    fl_balanced = CostSensitiveFLEngine(num_classes=8, num_features=39, seed=42)
    coef_bal, inter_bal = fl_balanced.train_client_local(X_train, y_train, cost_sensitive=True, epochs=10)
    fl_balanced.update_global_model([coef_bal], [inter_bal])
    res_bal = fl_balanced.evaluate(X_test, y_test)

    print(f"{'Attack Category':12s} | {'Train':6s} | {'Std TPR':8s} | {'Bal TPR':8s} | {'Delta':7s}")
    print("-" * 55)

    train_counts = np.bincount(y_train, minlength=8)
    for c in range(8):
        cat_name = ATTACK_CATEGORIES[c]
        std_tpr = res_std['per_class_tpr'][c] * 100
        bal_tpr = res_bal['per_class_tpr'][c] * 100
        delta = bal_tpr - std_tpr
        sign = "+" if delta >= 0 else ""
        print(f"{cat_name:12s} | {train_counts[c]:6d} | {std_tpr:7.2f}% | {bal_tpr:7.2f}% | {sign}{delta:6.2f}%")

    print("-" * 55)
    macro_std_f1 = res_std['f1_macro'] * 100
    macro_bal_f1 = res_bal['f1_macro'] * 100
    f1_delta = macro_bal_f1 - macro_std_f1
    print(f"{'Macro F1':12s} | {'--':6s} | {macro_std_f1:7.2f}% | {macro_bal_f1:7.2f}% | +{f1_delta:5.2f}%")


if __name__ == "__main__":
    benchmark_class_imbalance()
