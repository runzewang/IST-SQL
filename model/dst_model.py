"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from . import torch_utils
import numpy as np
import xlwt
class DSTModel(torch.nn.Module):
    """
    """
    def __init__(self, input_size, token_to_id=None, dst_type='sql'):
        '''
        '''
        super().__init__()
        self.transform_weight_ = torch_utils.add_params((input_size*2, input_size), "label-tranform")
        self.act_func = lambda x: torch.sigmoid(x)
        self.transform_weight = lambda x: torch.mm(x, self.transform_weight_)
        self.dst_type = dst_type
        if self.dst_type == 'sql':
            self.token_to_id = token_to_id

    def forward(self, state, dst_state, dst_value):
        '''
        state: dim
        dst_state: dst_num, dim
        dst_value: label_num, dim
        '''
        assert state.size(0) == dst_state.size(1) == dst_value.size(1)
        self.label_num = dst_value.size(0)
        state_patten = state.squeeze(0).expand(dst_state.size())
        stt_tem = self.transform_weight(torch.cat([state_patten, dst_state], dim=1))
        scores = torch.mm(stt_tem, dst_value.permute(1, 0)) # dst_num, label_num
        self.scores = scores
        return scores
    def bce_loss(self, dst_labels):
        '''
        dst_labels: [[labels]]
        '''
        dst_num = len(dst_labels)
        res = torch.FloatTensor(dst_num, self.label_num).cuda().fill_(0.)
        # print(res.size())
        if self.dst_type == 'sql':
            for dst_idx, labels in enumerate(dst_labels):
                for label in labels:
                    if label != 'None':
                        res[dst_idx, self.token_to_id[label]] = 1.
        elif self.dst_type == 'keywords':
            for key_idx, labels in enumerate(dst_labels):
                for label in labels:
                    if label != 100000:
                        res[key_idx, label] = 1.
        else:
            pass
        assert res.size() == self.scores.size()
        loss = torch.nn.BCEWithLogitsLoss()(self.scores, res)
        return loss