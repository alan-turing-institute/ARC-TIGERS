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
    "acc_sampler": AccSampler,
    "ig_sampler": InformationGainSampler,
    "isolation_forest": IsolationForestSampler,
    "minority_class": MinorityClassSampler,
}
