#!/usr/bin/env python3
"""
Print and plot top-N features by mean absolute SHAP value for each class.

For a LinearSVC, SHAP values are exact:
    SHAP(j, k, i) = coef[i][j] * (x[k,j] - E[X[:,j]])
    mean|SHAP(j,i)| = |coef[i][j]| * MAD[j]

For binary features: MAD[j] = 2 * freq[j] * (1 - freq[j])
This avoids materializing the dense matrix (would be ~14 GB here).

Usage:
    python3 get_features.py [data_file] [top_n]

    data_file — .cod.cl file used to compute feature frequencies (default: train.cod.cl)
    top_n     — number of features to show per class (default: 20)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import load


def load_data(path):
    features, labels = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            labels.append(parts[0])
            features.append({feat: 1 for feat in parts[1:]})
    return features, labels


def compute_shap(model, vectorizer, data_file):
    features, labels = load_data(data_file)
    X = vectorizer.transform(features)
    feature_names = np.array(vectorizer.get_feature_names_out())

    freq = np.asarray(X.mean(axis=0)).flatten()
    mad  = 2 * freq * (1 - freq)

    shap_per_class = {}
    for i, cls in enumerate(model.classes_):
        shap_per_class[cls] = np.abs(model.coef_[i]) * mad

    return shap_per_class, feature_names


def get_features(model, vectorizer, data_file="train.cod.cl", top_n=20):
    shap_per_class, feature_names = compute_shap(model, vectorizer, data_file)

    for cls, mean_abs_shap in shap_per_class.items():
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        print(f"\n=== Class: {cls} ===")
        print(f"  {'Feature':<40} {'Mean |SHAP|':>12}")
        print(f"  {'-'*40} {'-'*12}")
        for idx in top_idx:
            print(f"  {feature_names[idx]:<40} {mean_abs_shap[idx]:>12.6f}")

    plot_shap(shap_per_class, feature_names, top_n)


def plot_shap(shap_per_class, feature_names, top_n=20):
    classes = list(shap_per_class.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 0.4 * top_n + 2))
    if n_classes == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, cls, color in zip(axes, classes, colors):
        vals = shap_per_class[cls]
        top_idx = np.argsort(vals)[::-1][:top_n]
        top_vals  = vals[top_idx][::-1]
        top_names = feature_names[top_idx][::-1]

        ax.barh(range(top_n), top_vals, color=color, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_title(f"Class: {cls}", fontweight="bold")
        ax.set_xlabel("Mean |SHAP|")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Top features by mean |SHAP| value per class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_features.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to shap_features.png")
    plt.show()


if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "train.cod.cl"
    top_n     = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    model      = load("model.joblib")
    vectorizer = load("vectorizer.joblib")

    get_features(model, vectorizer, data_file, top_n)
