"""
https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
Custom samplers if data batch should be formed in a special way.
"""

import torch


class ExampleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return ExampleCustomBatch(batch)



