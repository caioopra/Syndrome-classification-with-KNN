from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix

import seaborn as sns

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer


def check_for_best_perplexity(X_train: np.ndarray):
    FILENAME = "perplexity_x_divergence.png"

    if Path(f"../reports/figures/{FILENAME}").exists():
        print(f"The file {FILENAME} already exists.")
        return

    # these valeus usually could be lower but, after some testing
    # with values greater than 100, there is a good reduction of the divergence
    perplexity = np.arange(140, 300, 5)

    divergence = []

    print("Searching for best perplexity to minimize the divergence...")
    for i in perplexity:
        tsne = TSNE(n_components=2, perplexity=i)
        X_tsne = tsne.fit_transform(X_train)
        divergence.append(tsne.kl_divergence_)

    plt.figure(figsize=(8, 6))
    plt.plot(perplexity, divergence, linestyle="-", marker="o")

    plt.xlabel("Perplexity")
    plt.ylabel("Divergence")
    plt.title("Perplexity x Divergence")

    print("Saving graph with analysis of perplexity x divergence of the dataset...")
    plt.savefig(f"../reports/figures/{FILENAME}")
    plt.close()


def create_tsne_visualizer(X: np.ndarray, X_train: np.ndarray):
    """Creates the t-SNE image for the visualization of the dataset with reduced dimensioons (2). This also creates a image to compare the reduction of the divergence varying"""
    check_for_best_perplexity(X_train)

    # value found after analysis of the graph
    PERPLEXITY = 280

    tsne = TSNE(n_components=2, random_state=42, perplexity=PERPLEXITY)

    X_tsne = tsne.fit_transform(X)

    print(f"KL Divergence of the TSNE: {tsne.kl_divergence_}")

    plt.figure(figsize=(8, 6))
    plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=X_tsne[:, 1], cmap="plasma")
    plt.colorbar()
    plt.title("t-SNE visualization of the dataset")
    plt.xlabel("First dimension of t-SNE")
    plt.ylabel("Second dimension of t-SNE")

    FILENAME = "tSNE_visualization.png"
    if Path(f"../reports/figures/{FILENAME}").exists():
        print(f"The file {FILENAME} already exists.")
        print("---" * 20)
        return

    print("Saving t-SNE visualization of the dataset...")
    print("---" * 20)
    plt.savefig(f"../reports/figures/{FILENAME}")
    plt.close()


def save_knn_accuracy(
    k_values: range, cv_scores_cosine: list, cv_scores_euclidean: list
):
    """Creates the images with KNN cross-validation accuracy for both cosine and euclidean distances with differente values of K"""
    COSINE_FILENAME = "knn_cosine_accuracy_number_of_neighbors.png"
    EUCLIDEAN_FILENAME = "knn_euclidean_accuracy_number_of_neighbors.png"

    if Path(f"../reports/figures/{COSINE_FILENAME}").exists():
        print(f"The file {COSINE_FILENAME} already exists.")
        print("---" * 20)
    else:
        print(f"Saving the figure {COSINE_FILENAME}...")
        plt.plot(list(k_values), cv_scores_cosine)
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Cross-validation accuracy")
        plt.title("KNN Classifier Cross-Validation Accuracy for Cosine distance")
        plt.xticks(k_values)
        plt.savefig(f"../reports/figures/{COSINE_FILENAME}")
        plt.close()

    if Path(f"../reports/figures/{EUCLIDEAN_FILENAME}").exists():
        print(f"The file {EUCLIDEAN_FILENAME} already exists.")
        print("---" * 20)
        return

    print(f"Saving the figure {EUCLIDEAN_FILENAME}...")
    plt.plot(list(k_values), cv_scores_euclidean)
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Cross-validation accuracy")
    plt.title("KNN Classifier Cross-Validation Accuracy for Euclidean distance")
    plt.xticks(k_values)
    plt.savefig(f"../reports/figures/{EUCLIDEAN_FILENAME}")
    plt.close()


def save_roc_auc(y_test_bin, y_train, y_pred_prob_cosine, y_pred_prob_euclidean):
    """Creates and sabe the image with the ROC AUC for cosine and euclidean"""
    FILENAME = "roc_auc.png"

    if Path(f"../reports/figures/{FILENAME}").exists():
        print(f"The file {FILENAME} already exists.")
        print("---" * 20)
        return

    print(f"Saving the figure {FILENAME}")

    plt.figure(figsize=(10, 10))
    for i in range(len(np.unique(y_train))):
        fpr_cosine, tpr_cosine, _ = roc_curve(y_test_bin.ravel(), y_pred_prob_cosine)
        roc_auc_1 = auc(fpr_cosine, tpr_cosine)
        plt.plot(
            fpr_cosine,
            tpr_cosine,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc_1:0.2f}) Cosine KNN (Class {i})",
        )

        fpr_euclidean, tpr_euclidean, _ = roc_curve(
            y_test_bin.ravel(), y_pred_prob_euclidean
        )
        roc_auc_1 = auc(fpr_euclidean, tpr_euclidean)
        plt.plot(
            fpr_euclidean,
            tpr_euclidean,
            lw=2,
            color="darkblue",
            label=f"ROC curve (AUC = {roc_auc_1:0.2f}) Euclidean KNN (Class {i})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for cosine and euclidean KNN")
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, 0), ncol=2)
    plt.savefig(f"../reports/figures/{FILENAME}")
    plt.close()

    print("---" * 20)


def create_confusion_matrices(X_test, y_test, best_knn_cosine, best_knn_euclidean):
    COSINE_FILENAME = "confusion_matrix_cosine.png"
    EUCLIDEAN_FILENAME = "confusion_matrix_euclidean.png"

    if Path(f"../reports/figures/{COSINE_FILENAME}").exists():
        print(f"The file {COSINE_FILENAME} already exists.")
        print("---" * 20)
    else:
        print(f"Saving the figure {COSINE_FILENAME}")

        cosine_preds = best_knn_cosine.predict(X_test)

        cm = confusion_matrix(y_test, cosine_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix for Cosine KNN")
        plt.savefig(f"../reports/figures/{COSINE_FILENAME}")

    if Path(f"../reports/figures/{EUCLIDEAN_FILENAME}").exists():
        print(f"The file {EUCLIDEAN_FILENAME} already exists.")
        print("---" * 20)
        return

    print(f"Saving the figure {EUCLIDEAN_FILENAME}")

    euclidean_preds = best_knn_euclidean.predict(X_test)

    cm = confusion_matrix(y_test, euclidean_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Euclidean KNN")
    plt.savefig(f"../reports/figures/{EUCLIDEAN_FILENAME}")
    print("---" * 20)


def create_report(ALL_COLUMNS, cosine_metrics, euclidean_metrics):
    """Creates the report with information about the cosine and euclidean metrics and the KNN models using both algorithms"""
    PDF_FILENAME = "cosine_euclidean_metrics.pdf"
    doc = SimpleDocTemplate(f"../reports/{PDF_FILENAME}", pagesize=letter)

    header_text = [["Cosine and Euclidean distance metrics"]]

    header_table = Table(header_text)

    # table header styling
    header_table_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 15),
        ]
    )
    header_table.setStyle(header_table_style)

    table = Table([ALL_COLUMNS, cosine_metrics, euclidean_metrics])

    # main table styles
    table_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.grey,
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.whitesmoke,
            ),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ]
    )
    table.setStyle(table_style)

    elements = [header_table, Spacer(0, 10), table]

    doc.build(elements)

    print(f"PDF created successfully: {PDF_FILENAME}")
