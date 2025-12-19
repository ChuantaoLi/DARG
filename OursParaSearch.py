# -*- coding: utf-8 -*-
import warnings
import traceback
import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict
from mlxtend.evaluate import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin


class Mutual_Nearest_Neighbors(object):
    def __init__(self, X: np.array, y: np.array):
        self.target: np.array = y
        self.data: np.array = X
        self.knn = {}
        self.nan = {}
        self.nan_edges = set()
        self.nan_num = {}

    def _initialize_structures(self):
        num_samples = len(self.data)
        self.knn = {j: set() for j in range(num_samples)}
        self.nan = {j: set() for j in range(num_samples)}
        self.nan_num = {j: 0 for j in range(num_samples)}
        self.nan_edges = set()

    def algorithm(self, k_selector: int = 5):
        self._initialize_structures()
        n_samples = self.data.shape[0]
        k_sel_safe = min(k_selector, n_samples - 1)
        if k_sel_safe <= 0:
            return

        nn = NearestNeighbors(n_neighbors=k_sel_safe + 1).fit(self.data)
        distances, _ = nn.kneighbors(self.data)
        sigmas = distances[:, k_sel_safe]

        tree = KDTree(self.data)
        all_knn_indices = tree.query_radius(self.data, r=sigmas)

        for i in range(n_samples):
            neighbors = set(all_knn_indices[i])
            neighbors.discard(i)
            self.knn[i] = neighbors

        for i in range(n_samples):
            for neighbor in self.knn[i]:
                neighbor = int(neighbor)
                if i in self.knn.get(neighbor, set()) and (i, neighbor) not in self.nan_edges:
                    self.nan[i].add(neighbor)
                    self.nan[neighbor].add(i)
                    self.nan_edges.add((i, neighbor))
                    self.nan_edges.add((neighbor, i))

        for i in range(len(self.data)):
            self.nan_num[i] = len(self.nan[i])


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def geometric_mean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    if np.any(recalls == 0):
        return 0.0

    g_mean = np.exp(np.mean(np.log(recalls)))
    return g_mean


@dataclass
class AdaptiveSamplingAdaBoost(BaseEstimator, ClassifierMixin):
    n_estimators: int = 50
    base_estimator_max_depth: int = 5
    mnn_k_selector: int = 5
    density_threshold: float = 0.5
    random_state: int = 42

    def __post_init__(self):
        self.base_estimators_: List[DecisionTreeClassifier] = []
        self.beta_: List[float] = []
        self.classes_: np.ndarray = None
        self.le_: LabelEncoder = None

    def _density_factor(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mnn = Mutual_Nearest_Neighbors(X=X, y=y)
        mnn.algorithm(k_selector=self.mnn_k_selector)

        n_samples = X.shape[0]
        rho_prime = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            rho_prime[i] = mnn.nan_num.get(i, 0)

        mn, mx = rho_prime.min(), rho_prime.max()
        if mx - mn < 1e-12:
            rho = np.zeros_like(rho_prime)
        else:
            rho = (rho_prime - mn) / (mx - mn)
        return rho

    def _confidence_factor(self, proba: np.ndarray, y_true_idx: np.ndarray) -> np.ndarray:
        p_true = proba[np.arange(proba.shape[0]), y_true_idx]
        proba_no_true = proba.copy()
        proba_no_true[np.arange(proba.shape[0]), y_true_idx] = -np.inf
        p_other_max = np.max(proba_no_true, axis=1)
        H = (1.0 - p_true + p_other_max) / 2.0
        mu = np.mean(H)
        sigma = np.std(H)
        if sigma < 1e-12:
            sigma = 1e-12
        delta = 1.0 - np.exp(-((H - mu) ** 2) / (2.0 * sigma ** 2))
        return delta, H, mu, sigma

    @staticmethod
    def _compute_beta(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, sample_weight: np.ndarray) -> float:
        correct_mask = (y_true_idx == y_pred_idx)
        w_correct = sample_weight[correct_mask].sum()
        w_incorrect = sample_weight[~correct_mask].sum()
        w_correct = max(w_correct, 1e-12)
        w_incorrect = max(w_incorrect, 1e-12)
        beta = 0.5 * np.log(w_correct / w_incorrect)
        return float(beta)

    def baseleaner_factor_t(self, t: int) -> float:
        m = self.n_estimators
        if m == 1: return 1.0
        num = np.tan((m - t) / (m - 1) * (np.pi / 4.0))
        den = sum(np.tan((m - tt) / (m - 1) * (np.pi / 4.0)) for tt in range(1, m + 1))
        return float(num / (den + 1e-12))

    def _best_gmm(self, Xc: np.ndarray, max_comp: int = 5) -> GaussianMixture:
        max_k = min(max_comp, len(Xc))
        best_bic, best = np.inf, None
        rng = np.random.RandomState(self.random_state)

        for k in range(1, max_k + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=rng, n_init=2)
            gmm.fit(Xc)
            bic = gmm.bic(Xc)
            if bic < best_bic:
                best_bic, best = bic, gmm
        return best

    def _get_cumulative_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        scores = np.zeros((X.shape[0], n_classes), dtype=float)

        if not self.base_estimators_:
            return np.full((X.shape[0], n_classes), 1.0 / n_classes)

        for beta_t, clf in zip(self.beta_, self.base_estimators_):
            proba = clf.predict_proba(X)
            if proba.shape[1] != n_classes:
                full = np.full((len(X), n_classes), 1e-12)
                cols = clf.classes_
                for j, cj in enumerate(cols):
                    if cj < n_classes:
                        full[:, cj] = proba[:, j]
                proba = full

            proba_sum = proba.sum(axis=1, keepdims=True)
            proba = np.divide(proba, proba_sum, out=np.full_like(proba, 1.0 / n_classes), where=proba_sum > 1e-12)
            scores += beta_t * proba

        scores = np.clip(scores, 1e-12, None)
        scores_sum = scores.sum(axis=1, keepdims=True)
        scores = np.divide(scores, scores_sum, out=np.full_like(scores, 1.0 / n_classes), where=scores_sum > 1e-12)
        return scores

    def _dynamic_sampling_round(self, X: np.ndarray, y_idx: np.ndarray,
                                proba: np.ndarray, delta: np.ndarray,
                                H: np.ndarray, muH: float, sigmaH: float,
                                rho: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state + t)
        classes, counts = np.unique(y_idx, return_counts=True)
        max_class_idx = np.argmax(counts)
        max_class = classes[max_class_idx]
        max_count = counts[max_class_idx]

        X_new, y_new = [], []
        remove_indices = set()

        for c in classes:
            if c == max_class:
                continue

            idx_c = np.where(y_idx == c)[0]
            Xc = X[idx_c]
            Hc, rhoc = H[idx_c], rho[idx_c]

            current_class_count = len(Xc)
            if current_class_count < 2: continue

            gmm = self._best_gmm(Xc, max_comp=5)
            clusters = gmm.predict(Xc)
            unique_clusters = np.unique(clusters)

            rho_avg = {s: rhoc[clusters == s].mean() if np.any(clusters == s) else 0.0 for s in unique_clusters}
            sizes = {s: np.sum(clusters == s) for s in unique_clusters}
            denom = sum((rho_avg[s] ** 1.0) * sizes[s] for s in unique_clusters) + 1e-12
            P = {s: ((rho_avg[s] ** 1.0) * sizes[s]) / denom for s in unique_clusters}

            need_total = int(max_count - current_class_count)
            if need_total <= 0: continue
            Ot = self.baseleaner_factor_t(t)

            for s in unique_clusters:
                target_s = int(np.round(P[s] * Ot * need_total))
                if target_s <= 0: continue

                in_s = np.where(clusters == s)[0]
                if len(in_s) < 2: continue

                S1 = in_s[(rhoc[in_s] >= self.density_threshold)]
                boundary_lo = float(muH - sigmaH)
                S2_S3_mask = rhoc[in_s] < self.density_threshold
                S2_S3_indices = in_s[S2_S3_mask]

                if len(S2_S3_indices) == 0: continue

                S2 = S2_S3_indices[Hc[S2_S3_indices] > boundary_lo]
                S3 = S2_S3_indices[Hc[S2_S3_indices] <= boundary_lo]

                if len(S3) > 0:
                    n_remove = max(0, int(0.1 * len(S3)))
                    if n_remove > 0:
                        rm_idx = rng.choice(S3, size=n_remove, replace=False)
                        for rid in rm_idx:
                            remove_indices.add(idx_c[rid])

                if len(S2) >= 1 and len(S1) >= 1:
                    for _ in range(target_s):
                        a_idx = rng.choice(S2)
                        b_idx = rng.choice(S1)
                        Xa, Xb = Xc[a_idx], Xc[b_idx]
                        alpha = rng.rand()
                        Xsyn = Xa + alpha * (Xb - Xa)
                        X_new.append(Xsyn)
                        y_new.append(c)
                elif len(in_s) >= 2:
                    if len(S2) >= 2:
                        for _ in range(target_s):
                            a, b = rng.choice(S2, size=2, replace=False)
                            Xa, Xb = Xc[a], Xc[b]
                            alpha = rng.rand()
                            Xsyn = Xa + alpha * (Xb - Xa)
                            X_new.append(Xsyn)
                            y_new.append(c)
                    else:
                        for _ in range(target_s):
                            a, b = rng.choice(in_s, size=2, replace=False)
                            Xa, Xb = Xc[a], Xc[b]
                            alpha = rng.rand()
                            Xsyn = Xa + alpha * (Xb - Xa)
                            X_new.append(Xsyn)
                            y_new.append(c)

        if X_new:
            X_aug = np.vstack([X, np.vstack(X_new)])
            y_aug = np.hstack([y_idx, np.array(y_new, dtype=int)])
        else:
            X_aug, y_aug = X, y_idx

        if remove_indices:
            keep_mask = np.ones(len(X_aug), dtype=bool)
            keep_mask[list(remove_indices)] = False
            X_aug = X_aug[keep_mask]
            y_aug = y_aug[keep_mask]

        return X_aug, y_aug, np.array(list(remove_indices), dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n = X.shape[0]
        w = np.ones(n, dtype=float) / n
        le = LabelEncoder().fit(y)
        y_idx = le.transform(y)
        self.le_ = le
        rng = np.random.RandomState(self.random_state)
        X_cur, y_cur = X.copy(), y_idx.copy()
        w_cur = w.copy()

        for t in range(1, self.n_estimators + 1):
            clf = DecisionTreeClassifier(max_depth=self.base_estimator_max_depth, random_state=rng.randint(0, 10 ** 9))

            if len(w_cur) < len(X_cur):
                extra = len(X_cur) - len(w_cur)
                non_zero_w_cur = w_cur[w_cur > 1e-12]
                mean_val = np.mean(non_zero_w_cur) if non_zero_w_cur.size > 0 else 1e-12
                w_cur = np.hstack([w_cur, np.full(extra, mean_val)])
                w_cur = w_cur / w_cur.sum()
            elif len(w_cur) > len(X_cur):
                w_cur = w_cur[:len(X_cur)]
                w_cur = w_cur / w_cur.sum()

            clf.fit(X_cur, y_cur, sample_weight=w_cur)
            y_pred_idx = clf.predict(X_cur)
            beta_t = self._compute_beta(y_cur, y_pred_idx, w_cur)

            self.base_estimators_.append(clf)
            self.beta_.append(beta_t)
            cumulative_proba = self._get_cumulative_proba(X_cur)
            delta, H, muH, sigmaH = self._confidence_factor(cumulative_proba, y_cur)

            rho = self._density_factor(X_cur, y_cur)
            adapt = np.exp(-delta * (1.0 - rho))
            correct = (y_pred_idx == y_cur).astype(float)
            w_next = adapt * w_cur * np.exp(-beta_t * correct)
            w_next = w_next + 1e-12
            w_next = w_next / w_next.sum()

            X_aug, y_aug, removed = self._dynamic_sampling_round(
                X_cur, y_cur, cumulative_proba, delta, H, muH, sigmaH, rho, t
            )
            X_cur, y_cur = X_aug, y_aug

            if len(w_next) < len(X_cur):
                extra = len(X_cur) - len(w_next)
                non_zero_w_next = w_next[w_next > 1e-12]
                mean_val = np.mean(non_zero_w_next) if non_zero_w_next.size > 0 else 1e-12
                w_next = np.hstack([w_next, np.full(extra, mean_val)])
            if len(w_next) > len(X_cur):
                w_next = w_next[:len(X_cur)]
            w_next = w_next / w_next.sum()
            w_cur = w_next
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._get_cumulative_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(pred_idx)


def load_data_file(filepath: str) -> Tuple[pd.DataFrame, pd.Series, str, float]:
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"    ERROR {filepath}: {e}")
        raise
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = X.apply(pd.to_numeric, errors='coerce')
    counts = Counter(y)
    class_dist = str(dict(counts))
    imbalance_ratio = 1.0
    if len(counts) > 1 and min(counts.values()) > 0:
        counts_values = counts.values()
        imbalance_ratio = max(counts_values) / min(counts_values)
    return X, y, class_dist, imbalance_ratio


def evaluate_model(model_instance, model_name: str,
                   X_train, y_train, X_test, y_test,
                   n_classes: int) -> Dict[str, float]:
    model = clone(model_instance)
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

    try:
        model.fit(X_train_np, y_train_np)
        y_pred = model.predict(X_test_np)

        acc = accuracy_score(y_test_np, y_pred)
        pre = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)
        gmean = geometric_mean_score(y_test_np, y_pred)

        auc = 0.0
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_np)
            if y_prob.shape[1] == len(model.classes_):
                try:
                    if n_classes == 2:
                        y_test_bin = (y_test_np == model.classes_[1]).astype(int)
                        auc = roc_auc_score(y_test_bin, y_prob[:, 1])
                    elif n_classes > 2:
                        lb = LabelBinarizer()
                        lb.fit(model.classes_)
                        y_test_bin = lb.transform(y_test_np)
                        if y_test_bin.shape[1] == y_prob.shape[1]:
                            auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')
                        elif y_test_bin.shape[1] == 1 and y_prob.shape[1] == 2:
                            auc = roc_auc_score(y_test_bin, y_prob[:, 1])
                        else:
                            auc = 0.0
                except ValueError:
                    auc = 0.0

        return {
            'Accuracy': round(acc, 3), 'Precision': round(pre, 3),
            'Recall': round(rec, 3), 'F1-Score': round(f1, 3),
            'G-Mean': round(gmean, 3), 'AUC': round(auc, 3)
        }

    except Exception as e:
        return {
            'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0,
            'F1-Score': 0.0, 'G-Mean': 0.0, 'AUC': 0.0
        }


def main():
    DATA_DIR = Path("Datasets")
    if not DATA_DIR.exists():
        return

    TEST_SIZE = 0.2
    RANDOM_STATE = 1024
    N_TRIALS = 50

    OUTPUT_FILE_DETAILED = "results_proposed_optuna_detailed.csv"
    OUTPUT_FILE_BEST = "results_proposed_optuna_best.csv"

    np.random.seed(RANDOM_STATE)

    all_detailed_trials = []
    all_best_results = []

    data_files = sorted(list(DATA_DIR.glob("*.csv")))
    if not data_files:
        return

    for data_file in data_files:
        print(f"\n{'=' * 60}")
        print(f"Doing: {data_file.name}")
        print(f"{'=' * 60}")

        try:
            X, y, class_dist, imbalance_ratio = load_data_file(str(data_file))
            mask = ~pd.isnull(X).any(axis=1)
            X, y = X[mask], y[mask]

            try:
                if np.all(y == y.astype(int)):
                    y = y.astype(int).astype(str)
                else:
                    y = y.astype(str)
            except:
                y = y.astype(str)

            if len(X) < 10: continue

            counts = pd.Series(y).value_counts()
            to_keep = counts[counts >= 2].index
            mask = pd.Series(y).isin(to_keep)
            X_filtered = X[mask]
            y_filtered = y[mask]

            if len(X_filtered) < 10 or len(np.unique(y_filtered)) < 2: continue

            n_classes = len(np.unique(y_filtered))

            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_filtered
            )

            print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {n_classes}")

            def objective(trial):
                k_sel = trial.suggest_int('mnn_k_selector', 2, 20)
                max_d = trial.suggest_int('base_estimator_max_depth', 1, 50)
                d_th = trial.suggest_float('density_threshold', 0.1, 0.9, step=0.05)

                model = AdaptiveSamplingAdaBoost(
                    n_estimators=50,
                    base_estimator_max_depth=max_d,
                    mnn_k_selector=k_sel,
                    density_threshold=d_th,
                    random_state=RANDOM_STATE
                )

                X_tr_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
                y_tr_np = y_train.values if isinstance(y_train, pd.Series) else y_train
                X_te_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                y_te_np = y_test.values if isinstance(y_test, pd.Series) else y_test

                try:
                    model.fit(X_tr_np, y_tr_np)
                    y_p = model.predict(X_te_np)
                    gm = geometric_mean_score(y_te_np, y_p)
                    return gm
                except Exception:
                    return 0.0

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

            with tqdm(total=N_TRIALS, desc=f"  Optimizing {data_file.name}", unit="trial") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)

                    best_val = study.best_value if study.best_value else 0.0
                    pbar.set_postfix({"Best G-Mean": f"{best_val:.4f}"})

                study.optimize(objective, n_trials=N_TRIALS, callbacks=[tqdm_callback])

            print(f"    Best G-Mean: {study.best_value:.4f}")
            print(f"    Best Params: {study.best_params}")

            trials_df = study.trials_dataframe()
            trials_df = trials_df.drop(columns=['datetime_start', 'datetime_complete', 'duration'], errors='ignore')
            trials_df['Dataset'] = data_file.name
            trials_df['Model'] = 'Ours_Adaptive_V5'
            all_detailed_trials.extend(trials_df.to_dict('records'))

            best_p = study.best_params
            best_model = AdaptiveSamplingAdaBoost(
                n_estimators=50,
                base_estimator_max_depth=best_p['base_estimator_max_depth'],
                mnn_k_selector=best_p['mnn_k_selector'],
                density_threshold=best_p['density_threshold'],
                random_state=RANDOM_STATE
            )

            final_metrics = evaluate_model(
                best_model, "Ours_Adaptive_V5",
                X_train, y_train, X_test, y_test,
                n_classes
            )

            result_row = {
                'Dataset': data_file.name,
                'n_Samples': len(X_filtered),
                'n_Features': X_filtered.shape[1],
                'n_Classes': n_classes,
                'Imbalance_Ratio': round(imbalance_ratio, 3),
                'Best_Params': str(best_p)
            }
            result_row.update(final_metrics)
            all_best_results.append(result_row)

        except Exception as e:
            traceback.print_exc()
            continue

    if all_detailed_trials:
        df_det = pd.DataFrame(all_detailed_trials)
        cols = list(df_det.columns)
        first = ['Dataset', 'Model', 'value', 'state']
        df_det = df_det[first + [c for c in cols if c not in first]]
        df_det.to_csv(OUTPUT_FILE_DETAILED, index=False)

    if all_best_results:
        df_best = pd.DataFrame(all_best_results)
        cols_order = [
            'Dataset', 'Best_Params', 'Accuracy', 'Precision', 'Recall',
            'F1-Score', 'G-Mean', 'AUC', 'n_Samples', 'n_Features',
            'n_Classes', 'Imbalance_Ratio'
        ]
        final_cols = [c for c in cols_order if c in df_best.columns]
        df_best = df_best[final_cols]
        df_best.to_csv(OUTPUT_FILE_BEST, index=False)


if __name__ == "__main__":

    main()
