from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from arc_tigers.sampling.metrics import softmax


class SSEPySampler:
    def __init__(
        self,
        seed: int,
        n_clusters: int,
        model_preds: np.ndarray,
        min_cluster_size: int = 10,
        **kwargs,
    ):
        self.rng = np.random.default_rng(seed)
        self.name = "ssepy"

        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.dataset_size = len(model_preds)
        self._cluster_labels = self._cluster(
            softmax(model_preds),
            n_clusters,
            min_cluster_size,
            self.rng.integers(int(1e9)),
        )

        self._observed_idx: list[int] = []
        self._sample_prob: list[float] = []

    @property
    def labelled_idx(self) -> list[int]:
        return self._observed_idx

    @property
    def unlabelled_idx(self) -> list[int]:
        return [i for i in range(self.dataset_size) if i not in self.labelled_idx]

    @property
    def sample_prob(self) -> list[float]:
        return self._sample_prob

    def sample(self, n_samples: int) -> tuple[list[int], list[float]]:
        cluster_n_samples, cluster_sizes = self._allocate_budget(n_samples)
        print(cluster_sizes)
        print(cluster_n_samples)
        observed_idx, sample_prob = self._draw_sample(cluster_n_samples, cluster_sizes)
        self._observed_idx = observed_idx.tolist()
        self._sample_prob = sample_prob.tolist()
        return self._observed_idx, self._sample_prob

    def _cluster(
        self,
        model_preds: np.ndarray,
        n_clusters: int,
        min_cluster_size: int,
        random_state: int,
    ) -> np.ndarray:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        X = model_preds
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Ensure X is 2D for clustering
        model = model.fit(X)
        cluster_labels = model.predict(X)

        # Check if any small clusters exist
        counts = Counter(cluster_labels)
        small = [lab for lab, cnt in counts.items() if cnt < min_cluster_size]
        if not small:
            return cluster_labels

        # Merge clusters with fewer than min_size samples into the nearest large cluster
        centers = model.cluster_centers_
        large = [lab for lab in counts if lab not in small]
        if not large:  # all labels in same cluster
            return np.zeros_like(cluster_labels)
        for lab in small:
            dists = np.linalg.norm(centers[large] - centers[lab], axis=1)
            nearest = large[np.argmin(dists)]
            cluster_labels[cluster_labels == lab] = nearest
        # Reindex strata labels so they remain consecutive integers
        uniq = np.unique(cluster_labels)
        mapping = {old: new for new, old in enumerate(uniq)}
        return np.vectorize(mapping.get)(cluster_labels)

    def _allocate_budget(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if n_samples > self._cluster_labels.size:
            msg = "Budget cannot exceed the number of samples."
            raise ValueError(msg)

        _, cluster_sizes = np.unique(self._cluster_labels, return_counts=True)

        cluster_n_samples = np.floor(
            n_samples * cluster_sizes / cluster_sizes.sum()
        ).astype(int)
        # If we still have leftover samples after allocation, distribute them among
        # strata that haven't reached their size limit.
        leftover = n_samples - cluster_n_samples.sum()
        eligible = np.where(cluster_n_samples < cluster_sizes)[0]
        while leftover > 0 and eligible.size > 0:
            empty_clusters = np.where(cluster_n_samples == 0)[0]
            if empty_clusters.size > 0:
                chosen = self.rng.choice(empty_clusters)
            else:
                chosen = self.rng.choice(eligible)
            cluster_n_samples[chosen] += 1
            leftover -= 1
            eligible = np.where(cluster_n_samples < cluster_sizes)[0]

        return cluster_n_samples, cluster_sizes

    def _draw_sample(
        self, cluster_n_samples: np.ndarray, cluster_sizes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(self.dataset_size)

        sampled_idx = []
        sampled_prob = []
        uniq, cluster_sizes = np.unique(self._cluster_labels, return_counts=True)
        # For each stratum, pick the predetermined number of samples at random
        for cluster_idx, (label, size) in enumerate(
            zip(uniq, cluster_sizes, strict=True)
        ):
            mask = self._cluster_labels == label
            idxs = indices[mask]
            n = cluster_n_samples[cluster_idx]
            sampled_idx.extend(self.rng.choice(idxs, size=n, replace=False).tolist())
            sampled_prob.extend([n / size] * n)

        return np.sort(sampled_idx), np.array(sampled_prob)
