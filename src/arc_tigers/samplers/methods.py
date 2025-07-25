from arc_tigers.samplers.acquisition import (
    AccSampler,
    DistanceSampler,
    InformationGainSampler,
    IsolationForestSampler,
    MinorityClassSampler,
)
from arc_tigers.samplers.fixed import FixedSampler
from arc_tigers.samplers.random import RandomSampler

SAMPLING_STRATEGIES = {
    "random": RandomSampler,
    "distance": DistanceSampler,
    "accuracy": AccSampler,
    "info_gain": InformationGainSampler,
    "isolation": IsolationForestSampler,
    "minority": MinorityClassSampler,
    "fixed": FixedSampler,
}
