import os
import numpy as np
import matplotlib.pyplot as plt

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def generate_artifacts(splits, model, meta, out_dir, task, model_type):
    artifacts = []

    # Prediction vs truth (test)
    y_true = splits["y_test"]
    y_pred = _predict(model, splits["X_test"], meta, task)
    if task == "regression":
        fig_path = os.path.join(out_dir, "pred_vs_true.png")
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Prediction vs True")
        _savefig(fig_path)
        artifacts.append(fig_path)
    else:
        # confusion-like scatter (binary/multiclass)
        fig_path = os.path.join(out_dir, "prob_or_label_hist.png")
        plt.figure()
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(splits["X_test"])
            if proba.ndim == 2 and proba.shape[1] > 1:
                for i in range(proba.shape[1]):
                    plt.hist(proba[:, i], bins=20, alpha=0.5, label=f"class {i}")
                plt.legend()
                plt.title("Predicted Probabilities")
            else:
                plt.hist(proba, bins=20)
                plt.title("Predicted Probability")
        else:
            plt.hist(y_pred, bins=20)
            plt.title("Predicted Labels")
        _savefig(fig_path)
        artifacts.append(fig_path)

    # Error distribution
    err_path = os.path.join(out_dir, "errors.png")
    plt.figure()
    if task == "regression":
        errors = y_pred - y_true
        plt.hist(errors, bins=30)
        plt.title("Error Distribution")
        _savefig(err_path)
    else:
        diff = (y_pred != y_true).astype(int)
        plt.hist(diff, bins=3)
        plt.title("Classification Misses (0=correct,1=wrong)")
        _savefig(err_path)
    artifacts.append(err_path)

    # Feature importance if available
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        idx = np.argsort(-fi)[:30]
        labels = []
        if "preprocess" in meta and meta["preprocess"] != None:
            # OneHot expands; try to get names
            try:
                labels = meta["preprocess"].get_feature_names_out()
                labels = labels[idx]
            except Exception:
                labels = [str(i) for i in idx]
        else:
            labels = [str(i) for i in idx]
        fi_path = os.path.join(out_dir, "feature_importance.png")
        plt.figure(figsize=(7, max(3, len(idx) * 0.25)))
        plt.barh(range(len(idx)), fi[idx])
        plt.yticks(range(len(idx)), labels)
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        _savefig(fi_path)
        artifacts.append(fi_path)

    return artifacts

def _predict(model, X, meta, task):
    # For torch models in nn/transformer we wrap a sklearn-like predict
    if hasattr(model, "predict"):
        return model.predict(X)
    elif hasattr(model, "predict_sklearn_like"):
        return model.predict_sklearn_like(X, task, meta)
    else:
        raise ValueError("Model has no predict method")
