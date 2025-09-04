from arc_tigers.samplers.acquisition import (
    AccSampler,
    DistanceSampler,
    EntropySampler,
    InformationGainSampler,
    IsolationForestSampler,
    MinorityClassSampler,
)
from arc_tigers.samplers.fixed import FixedSampler
from arc_tigers.samplers.random import RandomSampler
from arc_tigers.samplers.ssepy import SSEPySampler

SAMPLING_STRATEGIES = {
    "random": RandomSampler,
    "distance": DistanceSampler,
    "accuracy": AccSampler,
    "info_gain": InformationGainSampler,
    "isolation": IsolationForestSampler,
    "minority": MinorityClassSampler,
    "entropy": EntropySampler,
    "fixed": FixedSampler,
    "ssepy": SSEPySampler,
}
