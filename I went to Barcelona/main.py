import argparse
import numpy as np
import os

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn

np.random.seed(100)

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
            print("\n🔍 Running cross-validation to find best K for KNN...")
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            k_range = range(1, 11)
            best_k = None
            best_acc = -1

            for k in k_range:
                accs = []
                for train_idx, val_idx in kf.split(xtrain):
                    xtr, xval = xtrain[train_idx], xtrain[val_idx]
                    ytr, yval = ytrain[train_idx], ytrain[val_idx]

                    knn_model = KNN(k=k)
                    knn_model.fit(xtr, ytr)
                    preds_val = knn_model.predict(xval)
                    acc = accuracy_fn(preds_val, yval)
                    accs.append(acc)

                avg_acc = np.mean(accs)
                print(f"k = {k}: average val accuracy = {avg_acc:.2f}%")

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_k = k

            print(f"\n✅ Best K = {best_k} with accuracy = {best_acc:.2f}%\n")
            args.K = best_k

        method_obj = KNN(k=args.K)

    elif args.method == "logistic_regression":
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "kmeans":
        method_obj = KMeans(max_iters=args.max_iters)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 4. Train and evaluate
    preds_train = method_obj.fit(xtrain, ytrain)
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

    args = parser.parse_args()
    main(args)
