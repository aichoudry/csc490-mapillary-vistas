from enum import Enum
import torch.nn.functional as F

class LossFunction(Enum):
    CROSS_ENTROPY = F.cross_entropy
