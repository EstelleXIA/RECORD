import matplotlib.pyplot as plt
import torch.nn
from sklearn.metrics import auc, roc_curve
from sklearn import metrics as sk_metrics
import numpy as np
from itertools import cycle
from torch.nn import functional
from typing import Optional


def plot_fn(fpr, tpr, roc_auc, classes, figure_name=None):
    n_classes = len(classes)
    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic(ROC)")
    plt.legend(loc="lower right")
    if figure_name is not None:
        plt.savefig(figure_name)
    # plt.show()


def plot_roc_curve(y_test, y_score, classes, plot=False, figure_name=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        # fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # if plot:
    plot_fn(fpr, tpr, roc_auc, classes, figure_name)
    for old_key, new_key in enumerate(classes):
        roc_auc[new_key] = roc_auc.pop(old_key)
    for old_key, new_key in enumerate(classes):
        fpr[new_key] = fpr.pop(old_key)
        tpr[new_key] = tpr.pop(old_key)
    fpr = {key: value.tolist() for key, value in fpr.items()}
    tpr = {key: value.tolist() for key, value in tpr.items()}
    roc_auc.update({"fpr": fpr, "tpr": tpr})
    return roc_auc


def plot_comparison(reports, plot=True):
    fprs = [value["fpr"] for report, value in reports.items()]
    tprs = [value["tpr"] for report, value in reports.items()]
    roc_aucs = [value for report, value in reports.items()]
    models = [report for report, value in reports.items()]
    # assert len(fprs) == len(tprs) == len(roc_aucs) == len(models)
    figure, axes = plt.subplots(2, 3)
    axes = axes.tolist()[0] + axes.tolist()[1]
    for j, class_name in enumerate(fprs[0].keys()):
        axes[j].set_title(class_name)
        axes[j].set_xlabel("False Positive Rate")
        axes[j].set_ylabel("True Positive Rate")
        for i, model_name in enumerate(models):
            axes[j].plot(fprs[i][class_name],
                         tprs[i][class_name],
                         label=f"{class_name}-auc ROC curve ({roc_aucs[i][class_name]}:0.2f)",
                         # color="navy",
                         linestyle="--",
                         linewidth=1,)
    if plot:
        plt.show()
    return figure


def classification_report(gt_labels: torch.Tensor, gt_p: torch.Tensor,
                          p: torch.Tensor, plot: Optional[bool] = False, figure_name=None) -> dict:
    ce = torch.nn.CrossEntropyLoss()
    metrics = plot_roc_curve(functional.one_hot(gt_labels, 3).cpu().detach().numpy(), p.cpu().detach().numpy(),
                             ["PR", "SD", "PD"], figure_name=figure_name)

    metrics["ce_probability"] = ce(p.clip(min=1e-6).log(), gt_p).item()
    metrics["ce_discrete"] = ce(p.clip(min=1e-6).log(), gt_labels).item()

    metrics.update({
        "f1_micro": sk_metrics.f1_score(gt_labels.detach().cpu().numpy(),
                                        p.detach().cpu().argmax(1).numpy(), average="micro"),
        "f1_macro": sk_metrics.f1_score(gt_labels.detach().cpu().numpy(),
                                        p.detach().cpu().argmax(1).numpy(), average="macro"),
        "f1_weighted": sk_metrics.f1_score(gt_labels.detach().cpu().numpy(),
                                           p.detach().cpu().argmax(1).numpy(), average="weighted")
    })

    metrics["confusion_matrix"] = sk_metrics.confusion_matrix(gt_labels.detach().cpu().numpy(),
                                                              p.detach().cpu().argmax(1).numpy()).tolist()
    return metrics
