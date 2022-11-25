"""
    Training helper functions
    @author Qianyue He 
    @date 2022.4.24
"""

import torch
from copy import deepcopy

def mod_dict(info: dict, v, k = 'type') -> dict:
    result = deepcopy(info)
    result[k] = v
    return result

def saveModel(model, path_info: dict, other_stuff: dict = None, opt = None, amp = None):
    output_index = path_info['index'] % path_info['max_num'] + 1
    path = "%schkpt_%d_%s.%s"%(path_info['dir'], output_index, path_info['type'], path_info['ext'])
    checkpoint = {'model': model.state_dict(),}
    if not amp is None:
        checkpoint['amp'] =  amp.state_dict()
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    if not other_stuff is None:
        checkpoint.update(other_stuff)
    torch.save(checkpoint, path)

def nan_hook(self, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])