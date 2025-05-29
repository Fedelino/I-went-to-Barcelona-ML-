import argparse

import numpy as np
from torchinfo import summary
from itertools import product

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from sklearn.model_selection import train_test_split, StratifiedKFold

def run_with_args(args, data_train, data_val, label_train, label_val, n_classes):
    if args.nn_type == "mlp":
        data_train = data_train.reshape(data_train.shape[0], -1)
        data_val = data_val.reshape(data_val.shape[0], -1)
        means = data_train.mean(axis=0)
        stds = data_train.std(axis=0) + 1e-8
        data_train = normalize_fn(data_train, means, stds)
        data_val = normalize_fn(data_val, means, stds)
        model = MLP(input_size=data_train.shape[1], n_classes=n_classes, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    elif args.nn_type == "cnn":
        data_train = np.transpose(data_train, (0, 3, 1, 2))
        data_val = np.transpose(data_val, (0, 3, 1, 2))
        means = data_train.mean(axis=(0, 2, 3), keepdims=True)
        stds = data_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        data_train = normalize_fn(data_train, means, stds)
        data_val = normalize_fn(data_val, means, stds)
        model = CNN(input_channels=3, n_classes=n_classes)
    else:
        raise ValueError(f"Unsupported nn_type: {args.nn_type}")

    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    preds_train = method_obj.fit(data_train, label_train)
    preds_val = method_obj.predict(data_val)
    acc = accuracy_fn(preds_val, label_val)
    return acc

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    data_train, data_test, label_train, label_test = load_data()
    n_classes = get_n_classes(label_train)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    
    if args.grid:
        print("\nRunning grid search...")
        
        lrs = [1e-3, 1e-4, 1e-5]
        max_iters = [20, 100]
        hidden_dims = [128, 256, 512]
        num_layers_list = [2, 10]
        
        best_acc = 0
        best_config = None

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for lr, hd, nl, it in product(lrs, hidden_dims, num_layers_list, max_iters):
            args.lr = lr
            args.hidden_dim = hd
            args.num_layers = nl
            args.max_iters = it
            accs = []
            for train_idx, val_idx in skf.split(data_train, label_train):
                X_train, X_val = data_train[train_idx], data_train[val_idx]
                y_train, y_val = label_train[train_idx], label_train[val_idx]

                part_acc = run_with_args(args, X_train, X_val, y_train, y_val, n_classes)
                accs.append(part_acc)

            acc = np.mean(accs)
            print(f"lr={lr}, hidden_dim={hd}, num_layers={nl}, max_iters={it} => acc={acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_config = (lr, hd, nl, it)
        print(f"\n Best config: lr={best_config[0]}, hidden_dim={best_config[1]}, num_layers={best_config[2]}, max_iters={best_config[3]} => acc={best_acc:.4f}")
        return
    
    # Make a validation set
    if not args.test:
        data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=0.2, random_state=42) # x original, y validation
    else:
        data_val, label_val = data_test, label_test  # test mode: validate on test set
    # MOVED NORMALISATION TO STEP 3. THIS GOOD?



    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    if args.nn_type == "mlp":
        data_train = data_train.reshape(data_train.shape[0], -1)
        data_val = data_val.reshape(data_val.shape[0], -1)
        data_test = data_test.reshape(data_test.shape[0], -1)

        means = data_train.mean(axis=0)
        stds = data_train.std(axis=0) + 1e-8
        data_train = normalize_fn(data_train, means, stds)
        data_val = normalize_fn(data_val, means, stds)
        data_test = normalize_fn(data_test, means, stds)

        model = MLP(input_size=data_train.shape[1], n_classes=n_classes, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    elif args.nn_type == "cnn":
        # shape = (N, 28, 28, 3) → (N, 3, 28, 28)
        data_train = np.transpose(data_train, (0, 3, 1, 2))
        data_val = np.transpose(data_val, (0, 3, 1, 2))
        data_test = np.transpose(data_test, (0, 3, 1, 2))

        means = data_train.mean(axis=(0, 2, 3), keepdims=True)
        stds = data_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        data_train = normalize_fn(data_train, means, stds)
        data_val = normalize_fn(data_val, means, stds)
        data_test = normalize_fn(data_test, means, stds)
 
        model = CNN(input_channels=3, n_classes=n_classes)

    else:
        raise ValueError(f"Unsupported nn_type: {args.nn_type}")
        
    # already here, no code
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(data_train, label_train)

    # Predict on unseen data
    preds = method_obj.predict(data_val)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, label_train)
    macrof1 = macrof1_fn(preds_train, label_train)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, label_val)
    macrof1 = macrof1_fn(preds, label_val)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=20, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    
    parser.add_argument('--grid', action="store_true", help="Run grid search for best hyperparameters")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
