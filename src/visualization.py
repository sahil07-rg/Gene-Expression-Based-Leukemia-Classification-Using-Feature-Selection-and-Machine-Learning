import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def plot_roc_curve(roc_data, y_test):
    plt.figure(figsize=(8, 6))

    for name, probs in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f'{name} (AUC={auc_score:.3f})'
        )

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid()

    plt.show()
