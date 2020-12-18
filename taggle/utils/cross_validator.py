import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_kfold_cross_validator(df, target, n_splits=5, random_state=0):
    all_groups = pd.Series(df[target])
    if n_splits > 1:
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
        for idx_tr, idx_val in folds.split(all_groups, all_groups):
            train_df = df.iloc[idx_tr]
            valid_df = df.iloc[idx_val]
            print(len(train_df), len(valid_df))
            yield train_df, valid_df
    else:
        train_df, valid_df = train_test_split(
            df, random_state=random_state, stratify=df[target], test_size=0.1)
        yield train_df, valid_df


def stratified_group_kfold_cross_validator(df, target, group_target, n_splits=5, random_state=None):
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    groups = np.array(df[group_target].values)
    y = df[target].values
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(n_splits)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(random_state).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(n_splits):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(n_splits):
        train_groups = all_groups - groups_per_fold[i]
        valid_groups = groups_per_fold[i]

        train_idx = [i for i, g in enumerate(groups) if g in train_groups]
        valid_idx = [i for i, g in enumerate(groups) if g in valid_groups]
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        yield train_df, valid_df


def iterative_stratification_kfold_cross_validator(df, target, label_targets: list, n_splits=5, random_state=None):
    # required iterative-stratification

    X, y = df[label_targets].values, df[target].values

    # split data
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, random_state=random_state)

    for idx_tr, idx_val in mskf.split(X, y):
        train_df = df.iloc[idx_tr]
        valid_df = df.iloc[idx_val]
        print(len(train_df), len(valid_df))
        yield train_df, valid_df
