from typing import List

import numpy as np
import torch

def assign_data_owners_by_class(targets, class_to_owner_dict):
    per_sample_assignment = []
    for i in range(len(targets)):
        per_sample_assignment.append(class_to_owner_dict[targets[i].item()])

    return torch.Tensor(per_sample_assignment).int()
