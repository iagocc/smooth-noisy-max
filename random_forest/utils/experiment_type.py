from enum import Enum


class ExperimentType(Enum):
    DEFAULT = "default"
    DP = "dp"
    PF = "permute_nflip"
    SMOOTHED_DP = "smooth"
    RLNM_LAPLACE = "rlnm_laplace"
    RLNM_EXPONENTIAL = "rlnm_exponential"
    LOCAL_DAMPENING = "ldp"
    RLNM_T = "rlnm_t"
    RLNM_LLN = "rlnm_lln"
    RLNM_SLAP = "rlnm_smoothlap"
