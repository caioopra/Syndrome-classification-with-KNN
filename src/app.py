import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from scipy.spatial.distance import cosine, euclidean

from pickle import load

from visualization import (
    create_tsne_visualizer,
    save_knn_accuracy,
    save_roc_auc,
    create_confusion_matrices,
    create_report,
)


def load_syndromes() -> dict:
    """Loading the dictionary from the pickle file."""
    objects = []

    with open("../data/mini_gm_public_v0.1.p", "rb") as data:
        while True:
            try:
                objects.append(load(data))
            except EOFError:
                break
    syndromes = objects[0]

    return syndromes


def show_information_about_dataset(syndromes: dict) -> None:
    """Basic information about the dataset (amount of subjects, syndromes and images)"""
    syndromes_id = [syn_id for syn_id in syndromes.keys()]
    print(f"Syndromes IDs: {', '.join([syn_id for syn_id in syndromes_id])}\n")

    s = 0
    for syn_id in syndromes.keys():
        amount = len(syndromes[syn_id])
        s += amount

        print(f"Amount of subjects for syndrome {syn_id}: {amount}")

    print(f"\nTotal number of subjects: {s}")

    amount_of_images = 0
    for syn_id in syndromes_id:
        for subject in syndromes[syn_id]:
            for _ in syndromes[syn_id][subject].keys():
                amount_of_images += 1

    print("Total amount of images: ", amount_of_images)
    print("---" * 20)


def create_dataset(data) -> tuple[np.array, np.array]:
    syndromes_id = [syn_id for syn_id in data.keys()]
    X, y = [], np.array([])
    for syn_id in syndromes_id:
        for subject in data[syn_id]:
            for image in data[syn_id][subject].keys():
                X.append(data[syn_id][subject][image])
                y = np.append(y, syn_id)

    X = np.array(X)

    return X, y


def search_for_best_k(X_train, y_train) -> tuple:
    k_values = range(1, 31)
    cv_scores_cosine = []
    cv_scores_euclidean = []

    for k in k_values:
        knn_cosine = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_euclidean = KNeighborsClassifier(n_neighbors=k)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores_cosine = cross_val_score(
            knn_cosine, X_train, y_train, cv=kf, scoring="accuracy"
        )
        cv_scores_cosine.append(scores_cosine.mean())

        scores_euclidean = cross_val_score(
            knn_euclidean, X_train, y_train, cv=kf, scoring="accuracy"
        )
        cv_scores_euclidean.append(scores_euclidean.mean())

    save_knn_accuracy(k_values, cv_scores_cosine, cv_scores_euclidean)

    best_k_cosine = k_values[np.argmax(cv_scores_cosine)]
    best_k_euclidean = k_values[np.argmax(cv_scores_euclidean)]
    print("Best value of K for cosine distance: ", best_k_cosine)
    print("Best value of K for euclidena distance: ", best_k_euclidean)
    print("---" * 20)

    return best_k_cosine, best_k_euclidean


def calculate_metrics(X_test, y_test, best_knn_cosine, best_knn_euclidean):
    # calculates and prints f1 score, precision and recall for both models
    cosine_preds = best_knn_cosine.predict(X_test)
    cosine_f1_score = f1_score(y_test, cosine_preds, average="micro")
    print("Cosine f1 score: ", cosine_f1_score)

    euclidean_preds = best_knn_euclidean.predict(X_test)
    euclidean_f1_score = f1_score(y_test, euclidean_preds, average="micro")
    print("Euclidean f1 score: ", euclidean_f1_score)

    cosine_recall = recall_score(y_test, cosine_preds, average="micro")
    cosine_precision = precision_score(y_test, cosine_preds, average="micro")
    print(f"Cosine recall: {cosine_recall} | Cosine precision: {cosine_precision}")

    euclidean_recall = recall_score(y_test, euclidean_preds, average="micro")
    euclidean_precision = precision_score(y_test, euclidean_preds, average="micro")
    print(
        f"Euclidean recall: {euclidean_recall} | Euclidean precision: {euclidean_precision}"
    )
    print("---" * 20)

    cosine_metrics = {
        "f1_score": cosine_f1_score,
        "recall": cosine_recall,
        "precision": cosine_precision,
    }

    euclidean_metrics = {
        "f1_score": euclidean_f1_score,
        "recall": euclidean_recall,
        "precision": euclidean_precision,
    }

    return cosine_metrics, euclidean_metrics


def main():
    ALL_COLUMNS = [
        "Algorithm",
        "Best K",
        "F1 Score",
        "Precision Score",
        "Recall Score",
        "Average Distance",
        "Average Maximum Distance",
    ]

    syndromes = load_syndromes()

    show_information_about_dataset(syndromes)

    X, y = create_dataset(syndromes)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # TODO: assert that the dataset is correctly fetched and splited
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, shuffle=True
    )

    create_tsne_visualizer(X, X_train)

    """
    The next steps are related to the classification algorithm, using a 10 fold cross validation:
        - Calculating the cosine distance from each test set vector to the gallery vectors
        - Calculating the euclidian distance from each test set vector to the gallery vectors
        - Classifying each image to a syndrome ID using KNN algorithm (with both cosine and euclidian distances)
    """
    kfold = KFold(n_splits=10, shuffle=True)

    cosine_dist_avg = []
    cosine_all_max_dist = []
    euclidean_dist_avg = []
    euclidean_all_max_dist = []

    print("Calculating cosine and euclidian distance for 10 splits:")
    split_num = 1
    for _, test_index in kfold.split(X):
        X_test_split = X[test_index]

        cosine_distances = []
        euclidean_distances = []
        for test_vector in X_test_split:
            cosine_distances.append(
                [cosine(test_vector, gallery_vector) for gallery_vector in X]
            )
            euclidean_distances.append(
                [euclidean(test_vector, gallery_vector) for gallery_vector in X]
            )

        avg_cos_dist = np.mean(cosine_distances)
        cosine_dist_avg.append(avg_cos_dist)
        avg_euclidean_dist = np.mean(euclidean_distances)
        euclidean_dist_avg.append(avg_euclidean_dist)

        cosine_all_max_dist.append(np.max(cosine_distances))
        euclidean_all_max_dist.append(np.max(euclidean_distances))

        print(f"Split: {split_num}")
        print(
            f"Min. cosine distance: {np.min(cosine_distances)}; Max. cosine distance: {cosine_all_max_dist[-1]}"
        )
        print(
            f"Min. euclidean distance: {np.min(euclidean_distances)}; Max. euclidean distance: {euclidean_all_max_dist[-1]}"
        )
        print(f"Average cosine distance: {avg_cos_dist}")
        print(f"Average euclidean distance: {avg_euclidean_dist}\n")
        split_num += 1

    print(
        f"Final average for cosine distance: {np.mean(cosine_dist_avg)}\nFinal average for euclidean distance: {np.mean(euclidean_dist_avg)}"
    )
    print("---" * 20)

    best_k_cosine, best_k_euclidean = search_for_best_k(X_train, y_train)

    # testing both models with the best k value for each
    best_knn_cosine = KNeighborsClassifier(n_neighbors=best_k_cosine, metric="cosine")
    best_knn_cosine.fit(X_train, y_train)

    test_accuracy_cosine = best_knn_cosine.score(X_test, y_test)
    print("Test accuracy of cosine distance: ", test_accuracy_cosine)

    best_knn_euclidean = KNeighborsClassifier(
        n_neighbors=best_k_euclidean, metric="euclidean"
    )
    best_knn_euclidean.fit(X_train, y_train)

    test_accuracy_euclidean = best_knn_euclidean.score(X_test, y_test)
    print("Test accuracy of euclidean distance: ", test_accuracy_euclidean)
    print("---" * 20)

    # plotting ROC AUC
    # considering that the data has multiple classes, it is first necessary to convert it into a binary classification
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    best_knn_cosine_bin = KNeighborsClassifier(
        n_neighbors=best_k_cosine, metric="cosine"
    )
    best_knn_cosine_bin.fit(X_train, y_train_bin)

    best_knn_euclidean_bin = KNeighborsClassifier(n_neighbors=best_k_euclidean)
    best_knn_euclidean_bin.fit(X_train, y_train_bin)

    y_pred_prob_cosine = np.array(
        [
            pred[1]
            for prob_list in best_knn_cosine_bin.predict_proba(X_test)
            for pred in prob_list
        ]
    )
    y_pred_prob_euclidean = np.array(
        [
            pred[1]
            for prob_list in best_knn_euclidean_bin.predict_proba(X_test)
            for pred in prob_list
        ]
    )

    save_roc_auc(y_test_bin, y_train, y_pred_prob_cosine, y_pred_prob_euclidean)

    cosine_metrics, euclidean_metrics = calculate_metrics(
        X_test, y_test, best_knn_cosine, best_knn_euclidean
    )

    create_confusion_matrices(X_test, y_test, best_knn_cosine, best_knn_euclidean)

    # assembling information to create PDF with information about the two algorithms
    cosine_metrics = [
        "Cosine",
        best_k_cosine,
        f"{cosine_metrics['f1_score']:.3f}",
        f"{cosine_metrics['precision']:.3f}",
        f"{cosine_metrics['recall']:.3f}",
        f"{np.mean(cosine_dist_avg):.3f}",
        f"{np.mean(cosine_all_max_dist):.3f}",
    ]

    euclidean_metrics = [
        "Euclidean",
        best_k_euclidean,
        f"{euclidean_metrics['f1_score']:.3f}",
        f"{euclidean_metrics['precision']:.3f}",
        f"{euclidean_metrics['recall']:.3f}",
        f"{np.mean(euclidean_dist_avg):.3f}",
        f"{np.mean(euclidean_all_max_dist):.3f}",
    ]

    create_report(ALL_COLUMNS, cosine_metrics, euclidean_metrics)


if __name__ == "__main__":
    main()
