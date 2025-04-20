import argparse
import numpy as np
import os

from src.utils import manual_kfold_split
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn

np.random.seed(100)

"""
    elif args.method == "kmeans":
        best_score = -np.inf
        best_model = None
        best_k = None
        k_values = [args.n_clusters] if args.kmeans_k_range is None else list(map(int, args.kmeans_k_range.split(",")))

        if not args.test and args.cv:
            print("\nRunning cross-validation for KMeans...")
            folds = manual_kfold_split(xtrain, n_splits=5)
            for k in k_values:
                fold_scores = []
                for train_idx, val_idx in folds:
                    xtr, xval = xtrain[train_idx], xtrain[val_idx]
                    ytr, yval = ytrain[train_idx], ytrain[val_idx]

                    model = KMeans(max_iters=args.max_iters, n_init=args.kmeans_n_init,
                                   criterion=args.kmeans_scoring, n_clusters=k)
                    model.fit(xtr, ytr)
                    preds_val = model.predict(xval)
                    if args.kmeans_scoring == "accuracy":
                        score = accuracy_fn(preds_val, yval)
                    elif args.kmeans_scoring == "f1":
                        score = macrof1_fn(preds_val, yval)
                    else:
                        score = -np.sum((xval - model.centroids[model.predict(xval)])**2)
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)
                print(f"K = {k} → CV {args.kmeans_scoring} score = {avg_score:.4f}")
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_k = k

            print(f"\nBest KMeans K = {best_k} with CV {args.kmeans_scoring} = {best_score:.4f}")
            method_obj = best_model
            preds_train = best_model.fit(xtrain, ytrain)
        else:
            for k in k_values:
                print(f"\nTrying KMeans with K = {k}")
                model = KMeans(
                    max_iters=args.max_iters,
                    n_init=args.kmeans_n_init, 
                    criterion=args.kmeans_scoring,
                    n_clusters=k
                )
                preds_train = model.fit(xtrain, ytrain)

                if args.kmeans_scoring == "accuracy":
                    score = accuracy_fn(preds_train, ytrain)
                elif args.kmeans_scoring == "f1":
                    score = macrof1_fn(preds_train, ytrain)
                elif args.kmeans_scoring == "ssd":
                    score = -np.sum((xtrain - model.centroids[model.predict(xtrain)])**2)
                else:
                    raise ValueError("Invalid scoring criterion")

                print(f"K = {k} → {args.kmeans_scoring} score = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_k = k

            print(f"\n Best K = {best_k} with {args.kmeans_scoring} = {best_score:.4f}")
            method_obj = best_model
            preds_train = best_model.fit(xtrain, ytrain)
            """
            
def main(args):
    # 1. Load data
    if args.data_type == "features":
        feature_data = np.load("features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    # 2. Data preparation (validation split, normalization, bias term)
    if not args.test:
        pass  # Optional: Add your own validation split

    # 3. Method initialization
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    elif args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        if not args.test and args.cv:
            print("\nRunning manual cross-validation to find best K for KNN...")

            k_range = range(1, 50)
            best_k = None
            best_acc = -1

            folds = manual_kfold_split(xtrain, n_splits=5)

            for k in k_range:
                accs = []
                for train_idx, val_idx in folds:
                    xtr, xval = xtrain[train_idx], xtrain[val_idx]
                    ytr, yval = ytrain[train_idx], ytrain[val_idx]

                    knn_model = KNN(k=k)
                    knn_model.fit(xtr, ytr)
                    preds_val = knn_model.predict(xval)
                    acc = accuracy_fn(preds_val, yval)
                    accs.append(acc)

                avg_acc = np.mean(accs)
                print(f"K = {k}: Avg Val Accuracy = {avg_acc:.2f}%")

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_k = k

            print(f"\n Best K = {best_k} with Accuracy = {best_acc:.2f}%")
            args.K = best_k

        method_obj = KNN(k=args.K)
        preds_train = method_obj.fit(xtrain, ytrain)
        
    elif args.method == "logistic_regression":
        if not args.test and args.cv:
            print("\nRunning cross-validation for Logistic Regression hyperparameters...")
            lr_list = [1e-5, 1e-4, 1e-3, 1e-2]
            iter_list = [100, 200, 300]
            best_score = -np.inf
            best_lr = None
            best_iters = None
            folds = manual_kfold_split(xtrain, n_splits=5)

            for lr in lr_list:
                for max_iter in iter_list:
                    accs = []
                    for train_idx, val_idx in folds:
                        xtr, xval = xtrain[train_idx], xtrain[val_idx]
                        ytr, yval = ytrain[train_idx], ytrain[val_idx]

                        model = LogisticRegression(lr=lr, max_iters=max_iter)
                        model.fit(xtr, ytr)
                        preds_val = model.predict(xval)
                        accs.append(accuracy_fn(preds_val, yval))

                    avg_acc = np.mean(accs)
                    print(f"lr = {lr}, max_iters = {max_iter} → Avg Val Accuracy = {avg_acc:.4f}")
                    if avg_acc > best_score:
                        best_score = avg_acc
                        best_lr = lr
                        best_iters = max_iter

            print(f"\nBest Logistic Regression params: lr = {best_lr}, max_iters = {best_iters}, acc = {best_score:.4f}")
            args.lr = best_lr
            args.max_iters = best_iters

        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
        preds_train = method_obj.fit(xtrain, ytrain)
        
    elif args.method == "kmeans":
        print("\nRunning KMeans with fixed k = 8 and multiple initializations...")
        model = KMeans(
            max_iters=args.max_iters,
            n_init=args.kmeans_n_init,
            criterion=args.kmeans_scoring,
            n_clusters=8
        )
        preds_train = model.fit(xtrain, ytrain)
        method_obj = model


    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 4. Train and evaluate
    preds = method_obj.predict(xtest)

    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="dummy_classifier",
                        type=str, help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)")
    parser.add_argument("--data_path", default="data", type=str, help="path to your dataset")
    parser.add_argument("--data_type", default="features", type=str, help="features/original (MS2)")
    parser.add_argument("--K", type=int, default=1, help="number of neighbors for knn")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for iterative methods")
    parser.add_argument("--max_iters", type=int, default=100, help="maximum iterations for training")
    parser.add_argument("--test", action="store_true", help="evaluate on test set instead of validation")
    parser.add_argument("--nn_type", default="cnn", help="cnn or Transformer for MS2")
    parser.add_argument("--nn_batch_size", type=int, default=64, help="batch size for NN training")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation for KNN")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters for KMeans (defaults to number of classes)")
    parser.add_argument("--kmeans_n_init", type=int, default=10, help="Number of random initializations for KMeans")
    parser.add_argument("--kmeans_scoring", type=str, default="accuracy",
                        choices=["accuracy", "f1", "ssd"], help="Scoring method for KMeans")
    parser.add_argument("--kmeans_k_range", type=str, default=None,
                    help="Comma-separated list of cluster numbers to try, e.g., '3,4,5,6'")
    
    args = parser.parse_args()
    main(args)

