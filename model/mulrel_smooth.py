"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from . import torch_utils
import numpy as np
import xlwt
class MulRel(torch.nn.Module):
    """
    """
    def __init__(self, rel_num, dim_size, use_emb_w, src_none_node=False, tar_none_node=False, softmax_src=True, \
            tar_sum=True, use_tar_node=False, diagonal_matrix=False, use_normalize=False, score_use_normalize=False):
        '''
        mul_score_w, mul_emb_w: rel_num, dim
        src_none_node, tar_none_node: dim
        softmax_src: do softmax in src index
        tar_sum: whether sum in tar index
        use_tar_node: whether add tar_none_node into src
        diagonal_matrix: if score_w and emb_w are diagonal matrixes
        '''
        super().__init__()
        self.diagonal_matrix = diagonal_matrix
        if diagonal_matrix:
            self.mul_score_w = torch_utils.add_params((rel_num, dim_size), "mul_score_w")
        else:
            self.mul_score_w = torch_utils.add_params((rel_num, dim_size, dim_size), "mul_score_w")
        
        if use_emb_w:
            if diagonal_matrix:
                self.mul_emb_w = torch_utils.add_params((rel_num, dim_size), "mul_emb_w")
            else:
                self.mul_emb_w = torch_utils.add_params((rel_num, dim_size, dim_size), "mul_emb_w")
        else:
            self.mul_emb_w = None
        if src_none_node:
            self.src_none_node = torch_utils.add_params([dim_size], "src_none_node")
        else:
            self.src_none_node = None

        if tar_none_node:
            self.tar_none_node = torch_utils.add_params([dim_size], "tar_none_node")
        else:
            self.tar_none_node = None
        self.dim_size = dim_size
        self.rel_num = rel_num
        self.softmax_src = softmax_src
        self.tar_sum = tar_sum
        self.use_tar_node = use_tar_node
        self.use_normalize = use_normalize
        self.score_use_normalize = score_use_normalize

    def forward(self, src_emb, tar_emb, src_mask, tar_mask):
        '''
        src_emb: batchsize, src_num, dim
        tar_emb: batchsize, tar_num, dim
        src_mask, tar_mask: batchsize, src_num(tar_num)
        '''
        # print('src_emb.size()', src_emb.size())
        # print('tar_emb.size()', tar_emb.size())
        batchsize = src_emb.size(0)
        # print(self.dim_size)
        # print(src_emb.size(2))
        # print(tar_emb.size(2))
        assert self.dim_size == src_emb.size(2)
        if self.src_none_node is not None:
            none_node_mask = torch.ones(batchsize, 1).cuda()
            src_mask_ = torch.cat([src_mask, none_node_mask], dim=1)
            src_emb_ = torch.cat([src_emb, self.src_none_node.view(1, 1, -1).expand(batchsize, 1, src_emb.size(2))], dim=1)
        else:
            src_mask_ = src_mask
            src_emb_ = src_emb

        if self.tar_none_node is not None:
            none_node_mask = torch.ones(batchsize, 1).cuda()
            tar_mask_ = torch.cat([tar_mask, none_node_mask], dim=1)
            tar_emb_ = torch.cat([tar_emb, self.tar_none_node.view(1, 1, -1).expand(batchsize, 1, tar_emb.size(2))], dim=1)
        else:
            tar_mask_ = tar_mask
            tar_emb_ = tar_emb
        if self.score_use_normalize:
            mul_score_w_norm = F.normalize(self.mul_score_w)
        else:
            mul_score_w_norm = self.mul_score_w
        if self.diagonal_matrix:
            scores = src_emb_.unsqueeze(0) * mul_score_w_norm.view(mul_score_w_norm.size(0), 1, 1, -1) # rel_num, batch, src_num, dim
        else:
            scores = torch.matmul(src_emb_.unsqueeze(0).unsqueeze(3).expand(self.rel_num, batchsize, src_emb_.size(1), 1, src_emb_.size(2)), \
                mul_score_w_norm.view(self.rel_num, 1, 1, self.dim_size, self.dim_size)).squeeze(3) # rel_num, batch, src_num, dim
        # print(scores.size())
        # rel_num, batch, src_num, tar_num
        src_tar_score = torch.sum(scores.unsqueeze(3) * tar_emb_.view(1, tar_emb_.size(0), 1, tar_emb_.size(1), -1), dim=-1)
        
        src_tar_mask = src_mask_.unsqueeze(2) * tar_mask_.unsqueeze(1)
        src_tar_score = src_tar_score.add((1-src_tar_mask).mul(-1e10).unsqueeze(0)).mul(1 / np.sqrt(src_emb_.size(2)))
        
        if self.softmax_src:
            src_tar_weight = F.softmax(src_tar_score, dim=2) # rel_num, batch, src_num, tar_num
        else:
            src_tar_weight = F.softmax(src_tar_score, dim=3)

        if self.mul_emb_w is not None:
            if self.use_normalize:
                mul_emb_w_norm = F.normalize(self.mul_emb_w)
            else:
                mul_emb_w_norm = self.mul_emb_w
            if self.diagonal_matrix:
                rel_tar_emb = tar_emb_.unsqueeze(0) * mul_emb_w_norm.view(mul_emb_w_norm.size(0), 1, 1, -1) # rel, batch, tar, dim
            else:
                rel_tar_emb = torch.matmul(tar_emb_.unsqueeze(0).unsqueeze(3).expand(self.rel_num, batchsize, tar_emb_.size(1), 1, tar_emb_.size(2)), \
                mul_emb_w_norm.view(self.rel_num, 1, 1, self.dim_size, self.dim_size)).squeeze(3) # rel_num, batch, src_num, dim
        else:
            rel_tar_emb = tar_emb_.unsqueeze(0)
        
        if not self.use_tar_node:
            rel_tar_emb = rel_tar_emb[:, :, :-1, :]
            src_tar_weight = src_tar_weight[:, :, :, :-1]

        if self.tar_sum:
            # return : batch, src, dim
            if self.mul_emb_w is not None:
                if self.use_normalize:
                    mul_emb = torch.sum(torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=3), dim=0)
                else:
                    mul_emb = torch.sum(torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=3), dim=0).mul(1. / np.sqrt(self.mul_emb_w.size(0)))
            else:
                mul_emb = torch.sum(torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=3), dim=0)
            
            if self.src_none_node is not None:
                mul_emb = mul_emb[:, :-1, :]
        else:
            # return: batch, src, tar, dim
            if self.mul_emb_w is not None:
                if self.use_normalize:
                    mul_emb = torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=0)
                else:
                    mul_emb = torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=0).mul(1. / np.sqrt(self.mul_emb_w.size(0)))
            else:
                mul_emb = torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=0)
            if self.src_none_node is not None:
                mul_emb = mul_emb[:, :-1, :, :]
        return mul_emb
    def norm_loss(self, e):
        loss = 0.
        X = F.normalize(self.mul_score_w)
        diff = (X.view(X.size(0), 1, -1) - X.view(1, X.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
        # diff = diff * (diff < 1).float()
        # diff = diff * (diff < 1.5).float()
        diff = diff.mul(e)
        loss -= torch.sum(diff)
        if self.mul_emb_w is not None:
            X = F.normalize(self.mul_emb_w)
            diff = (X.view(X.size(0), 1, -1) - X.view(1, X.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
            # diff = diff * (diff < 1).float()
            # diff = diff * (diff < 1.5).float()
            diff = diff.mul(e)
            loss -= torch.sum(diff)
        return loss
    def eu_loss(self, e):
        loss = 0.
        X = self.mul_score_w
        diff = (X.view(X.size(0), 1, -1) - X.view(1, X.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
        diff = diff.mul(e)
        loss -= torch.sum(diff)
        if self.mul_emb_w is not None:
            X = self.mul_emb_w
            diff = (X.view(X.size(0), 1, -1) - X.view(1, X.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
            diff = diff.mul(e)
            loss -= torch.sum(diff)
        return loss
    def cosine_loss(self, e):
        loss = 0.
        rel_num, dim = self.mul_score_w.size()
        eye_matrix = torch.eye(rel_num).cuda()
        matrix_1 = self.mul_score_w.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = self.mul_score_w.unsqueeze(1).expand(rel_num, rel_num, dim)
        cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
        if rel_num != 1:
            average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
            loss -= average_score.mul(e)
        else:
            average_score = 0

        if self.mul_emb_w is not None:
            matrix_1 = self.mul_emb_w.unsqueeze(0).expand(rel_num, rel_num, dim)
            matrix_2 = self.mul_emb_w.unsqueeze(1).expand(rel_num, rel_num, dim)
            cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
            if rel_num != 1:
                average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
                loss -= average_score.mul(e)
            else:
                average_score = 0
        return loss
    def min_cosine_loss(self, e):
        loss = 0.
        rel_num, dim = self.mul_score_w.size()
        eye_matrix = torch.eye(rel_num).cuda()
        matrix_1 = self.mul_score_w.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = self.mul_score_w.unsqueeze(1).expand(rel_num, rel_num, dim)
        cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
        if rel_num != 1:
            average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
            loss += average_score.mul(e)
        else:
            average_score = 0

        if self.mul_emb_w is not None:
            matrix_1 = self.mul_emb_w.unsqueeze(0).expand(rel_num, rel_num, dim)
            matrix_2 = self.mul_emb_w.unsqueeze(1).expand(rel_num, rel_num, dim)
            cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
            if rel_num != 1:
                average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
                loss += average_score.mul(e)
            else:
                average_score = 0
        return loss
    def min_square_loss(self, e):
        loss = 0.
        rel_num, dim = self.mul_score_w.size()
        eye_matrix = torch.eye(rel_num).cuda()
        matrix_1 = self.mul_score_w.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = self.mul_score_w.unsqueeze(1).expand(rel_num, rel_num, dim)
        cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
        if rel_num != 1:
            average_score = torch.sum((cos_score * (1 - eye_matrix)).pow(2)) / (rel_num*rel_num-rel_num)
            loss += average_score.mul(e)
        else:
            average_score = 0

        if self.mul_emb_w is not None:
            matrix_1 = self.mul_emb_w.unsqueeze(0).expand(rel_num, rel_num, dim)
            matrix_2 = self.mul_emb_w.unsqueeze(1).expand(rel_num, rel_num, dim)
            cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
            if rel_num != 1:
                average_score = torch.sum((cos_score * (1 - eye_matrix)).pow(2)) / (rel_num*rel_num-rel_num)
                loss += average_score.mul(e)
            else:
                average_score = 0
        return loss
    def calculate_distance(self):
        score_weight_distance = {}
        emb_weight_distance = {}
        score_weight_distance['cosine_distance'] = self.cosine_distance(self.mul_score_w)
        score_weight_distance['standard_euclidean_distance'] = self.standard_euclidean_distance(self.mul_score_w)
        score_weight_distance['euclidean_distance'] = self.euclidean_distance(self.mul_score_w)
        score_weight_distance['average_variance'] = self.average_variance(self.mul_score_w)
        score_weight_distance['corr_distance'] = self.corr_distance(self.mul_score_w)
        score_weight_distance['dot_product_score'] = self.dot_product_score(self.mul_score_w)
        if self.mul_emb_w is not None:
            emb_weight_distance['cosine_distance'] = self.cosine_distance(self.mul_emb_w)
            emb_weight_distance['standard_euclidean_distance'] = self.standard_euclidean_distance(self.mul_emb_w)
            emb_weight_distance['euclidean_distance'] = self.euclidean_distance(self.mul_emb_w)
            emb_weight_distance['average_variance'] = self.average_variance(self.mul_emb_w)
            emb_weight_distance['corr_distance'] = self.corr_distance(self.mul_emb_w)
            emb_weight_distance['dot_product_score'] = self.dot_product_score(self.mul_emb_w)
        return score_weight_distance, emb_weight_distance
    def write_weight_distance(self, sheet):
        '''
        sheet is a sheet instance in xlwt package
        '''
        score_weight_distance, emb_weight_distance = self.calculate_distance()
        cl = 0
        ###########
        sheet.write(cl, 0, 'score_weight_distance')
        cl += 1
        sheet, cl = self.write_distance(score_weight_distance, cl, sheet)
        if self.mul_emb_w is not None:
            sheet.write(cl, 0, 'emb_weight_distance')
            cl += 1
            sheet, cl = self.write_distance(emb_weight_distance, cl, sheet)
        #########
        return sheet
    def write_distance(self, score_weight_distance, cl, sheet):
        cosine_distance, average_cosine_distance = score_weight_distance['cosine_distance']
        sheet.write(cl, 0, 'cosine_distance')
        cl += 1
        for i in range(cosine_distance.shape[0]):
            for j in range(cosine_distance.shape[1]):
                sheet.write(i+cl, j, cosine_distance[i, j].item())
        cl += cosine_distance.shape[0]
        sheet.write(cl, 0, 'average_cosine_distance')
        sheet.write(cl, 1, average_cosine_distance.item())
        cl += 1
        #######
        corr_distance, average_corr_distance = score_weight_distance['corr_distance']
        sheet.write(cl, 0, 'corr_distance')
        cl += 1
        for i in range(corr_distance.shape[0]):
            for j in range(corr_distance.shape[1]):
                sheet.write(i+cl, j, corr_distance[i, j].item())
        cl += corr_distance.shape[0]
        sheet.write(cl, 0, 'average_corr_distance')
        sheet.write(cl, 1, average_corr_distance.item())
        cl += 1
        #####################
        standard_euclidean_distance, average_standard_euclidean_distance = score_weight_distance['standard_euclidean_distance']
        sheet.write(cl, 0, 'standard_euclidean_distance')
        cl += 1
        for i in range(standard_euclidean_distance.shape[0]):
            for j in range(standard_euclidean_distance.shape[1]):
                sheet.write(i+cl, j, standard_euclidean_distance[i, j].item())
        cl += standard_euclidean_distance.shape[0]
        sheet.write(cl, 0, 'average_standard_euclidean_distance')
        sheet.write(cl, 1, average_standard_euclidean_distance.item())
        cl += 1
        #####################
        euclidean_distance, average_euclidean_distance = score_weight_distance['euclidean_distance']
        sheet.write(cl, 0, 'euclidean_distance')
        cl += 1
        for i in range(euclidean_distance.shape[0]):
            for j in range(euclidean_distance.shape[1]):
                sheet.write(i+cl, j, euclidean_distance[i, j].item())
        cl += euclidean_distance.shape[0]
        sheet.write(cl, 0, 'average_euclidean_distance')
        sheet.write(cl, 1, average_euclidean_distance.item())
        cl += 1
        #####################
        dot_product_score, average_dot_product_score = score_weight_distance['dot_product_score']
        sheet.write(cl, 0, 'dot_product_score')
        cl += 1
        for i in range(dot_product_score.shape[0]):
            for j in range(dot_product_score.shape[1]):
                sheet.write(i+cl, j, dot_product_score[i, j].item())
        cl += dot_product_score.shape[0]
        sheet.write(cl, 0, 'average_dot_product_score')
        sheet.write(cl, 1, average_dot_product_score.item())
        cl += 1
        #####################
        average_, variance_ = score_weight_distance['average_variance']
        sheet.write(cl, 0, 'average_variance')
        cl += 1
        for i in range(average_.shape[0]):
            sheet.write(cl, i, average_[i].item())
        cl += 1
        for i in range(variance_.shape[0]):
            sheet.write(cl, i, variance_[i].item())
        cl += 1
        return sheet, cl
        

    def cosine_distance(self, input_matrix):
        '''
        input matrix: (rel_num, vector)
        '''
        rel_num, dim = input_matrix.size()
        eye_matrix = torch.eye(rel_num).cuda()
        matrix_1 = input_matrix.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = input_matrix.unsqueeze(1).expand(rel_num, rel_num, dim)
        cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
        average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
        return cos_score.cpu().data.numpy(), average_score.cpu().data.numpy()
    def standard_euclidean_distance(self, input_matrix):
        X = F.normalize(input_matrix)
        diff = (X.view(X.size(0), 1, -1) - X.view(1, X.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
        average_score = torch.sum(diff) / 2.
        return diff.cpu().data.numpy(), average_score.cpu().data.numpy()
    def euclidean_distance(self, input_matrix):
        diff = (input_matrix.view(input_matrix.size(0), 1, -1) - input_matrix.view(1, input_matrix.size(0), -1)).pow(2).sum(dim=2).add_(1e-10).sqrt()
        average_score = torch.sum(diff) / 2.
        return diff.cpu().data.numpy(), average_score.cpu().data.numpy()
    def average_variance(self, input_matrix):
        average_data = torch.mean(input_matrix, dim=1)
        variance_data = torch.var(input_matrix, dim=1)
        return average_data.cpu().data.numpy(), variance_data.cpu().data.numpy()
    def corr_distance(self, input_matrix):
        rel_num, dim = input_matrix.size()
        eye_matrix = torch.eye(rel_num).cuda()
        mean_matrix = input_matrix - torch.mean(input_matrix, dim=1).view(rel_num, 1).expand(rel_num, dim)
        matrix_1 = mean_matrix.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = mean_matrix.unsqueeze(1).expand(rel_num, rel_num, dim)
        cos_score = torch.cosine_similarity(matrix_1, matrix_2, dim=2)
        average_score = torch.sum(cos_score * (1 - eye_matrix)) / (rel_num*rel_num-rel_num)
        return cos_score.cpu().data.numpy(), average_score.cpu().data.numpy()
    def dot_product_score(self, input_matrix):
        rel_num, dim = input_matrix.size()
        eye_matrix = torch.eye(rel_num).cuda()
        matrix_1 = input_matrix.unsqueeze(0).expand(rel_num, rel_num, dim)
        matrix_2 = input_matrix.unsqueeze(1).expand(rel_num, rel_num, dim)
        dot_score = torch.sum(matrix_1 * matrix_2, dim=2)
        dot_score = dot_score * (1 - eye_matrix)
        average_score = torch.sum(dot_score) / (rel_num*rel_num-rel_num)
        return dot_score, average_score