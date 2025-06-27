from arc_tigers.sample.acquisition import (
    AccSampler,
    DistanceSampler,
    InformationGainSampler,
    IsolationForestSampler,
    MinorityClassSampler,
)
from arc_tigers.sample.random import RandomSampler

SAMPLING_STRATEGIES = {
    "random": RandomSampler,
    "distance": DistanceSampler,
    "random_forest_acc": AccSampler,
    "random_forest_ig": InformationGainSampler,
    "iForest": IsolationForestSampler,
    "minority_class": MinorityClassSampler,
}
