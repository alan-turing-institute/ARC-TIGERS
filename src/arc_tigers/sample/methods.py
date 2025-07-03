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
    "accuracy": AccSampler,
    "info_gain": InformationGainSampler,
    "isolation": IsolationForestSampler,
    "minority": MinorityClassSampler,
}
