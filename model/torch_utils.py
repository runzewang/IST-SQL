"""Contains various utility functions for Dynet models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .decoder import flatten_distribution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_layer(exp, weights, biases=None):
    # exp: input as size_1 or 1 x size_1
    # weight: size_1 x size_2
    # bias: size_2
    if exp.dim() == 1:
        exp = torch.unsqueeze(exp, 0)
    assert exp.size()[1] == weights.size()[0]
    if biases is not None:
        assert weights.size()[1] == biases.size()[0]
        result = torch.mm(exp, weights) + biases
    else:
        result = torch.mm(exp, weights)
    return result


def compute_loss(gold_seq,
                 scores,
                 index_to_token_maps,
                 gold_tok_to_id,
                 noise=0.00000001):
    """ Computes the loss of a gold sequence given scores.

    Inputs:
        gold_seq (list of str): A sequence of gold tokens.
        scores (list of dy.Expression): Expressions representing the scores of
            potential output tokens for each token in gold_seq.
        index_to_token_maps (list of dict str->list of int): Maps from index in the
            sequence to a dictionary mapping from a string to a set of integers.
        gold_tok_to_id (lambda (str, str)->list of int): Maps from the gold token
            and some lookup function to the indices in the probability distribution
            where the gold token occurs.
        noise (float, optional): The amount of noise to add to the loss.

    Returns:
        dy.Expression representing the sum of losses over the sequence.
    """
    assert len(gold_seq) == len(scores) == len(index_to_token_maps)

    losses = []
    for i, gold_tok in enumerate(gold_seq):
        score = scores[i]
        token_map = index_to_token_maps[i]

        gold_indices = gold_tok_to_id(gold_tok, token_map)
        assert len(gold_indices) > 0
        noise_i = noise
        if len(gold_indices) == 1:
            noise_i = 0

        probdist = score
        # print('score', score)
        # print('gold_indices', gold_indices)
        prob_of_tok = noise_i + torch.sum(probdist[gold_indices])
        # print('prob_of_tok', prob_of_tok)
        losses.append(-torch.log(prob_of_tok))

    return torch.sum(torch.stack(losses))


def get_seq_from_scores(scores, index_to_token_maps):
    """Gets the argmax sequence from a set of scores.

    Inputs:
        scores (list of dy.Expression): Sequences of output scores.
        index_to_token_maps (list of list of str): For each output token, maps
            the index in the probability distribution to a string.

    Returns:
        list of str, representing the argmax sequence.
    """
    seq = []
    for score, tok_map in zip(scores, index_to_token_maps):
        # score_numpy_list = score.cpu().detach().numpy()
        score_numpy_list = score.cpu().data.numpy()
        assert score.size()[0] == len(tok_map) == len(list(score_numpy_list))
        seq.append(tok_map[np.argmax(score_numpy_list)])
    return seq
def get_seq_from_scores_flatten(scores, index_to_token_maps):
    """Gets the argmax sequence from a set of scores.

    Inputs:
        scores (list of dy.Expression): Sequences of output scores.
        index_to_token_maps (list of list of str): For each output token, maps
            the index in the probability distribution to a string.

    Returns:
        list of str, representing the argmax sequence.
    """
    seq = []
    for score, tok_map in zip(scores, index_to_token_maps):
        # score_numpy_list = score.cpu().detach().numpy()

        score_numpy_list = score.cpu().data.numpy().tolist()
        tok_map, score_numpy_list = flatten_distribution(tok_map, score_numpy_list)
        assert len(tok_map) == len(list(score_numpy_list))
        score_numpy_list = np.array(score_numpy_list)
        seq.append(tok_map[np.argmax(score_numpy_list)])
    return seq

def per_token_accuracy(gold_seq, pred_seq):
    """ Returns the per-token accuracy comparing two strings (recall).

    Inputs:
        gold_seq (list of str): A list of gold tokens.
        pred_seq (list of str): A list of predicted tokens.

    Returns:
        float, representing the accuracy.
    """
    num_correct = 0
    for i, gold_token in enumerate(gold_seq):
        if i < len(pred_seq) and pred_seq[i] == gold_token:
            num_correct += 1

    return float(num_correct) / len(gold_seq)

def forward_one_multilayer(rnns, lstm_input, layer_states, dropout_amount=0.):
    """ Goes forward for one multilayer RNN cell step.

    Inputs:
        lstm_input (dy.Expression): Some input to the step.
        layer_states (list of dy.RNNState): The states of each layer in the cell.
        dropout_amount (float, optional): The amount of dropout to apply, in
            between the layers.

    Returns:
        (list of dy.Expression, list of dy.Expression), dy.Expression, (list of dy.RNNSTate),
        representing (each layer's cell memory, each layer's cell hidden state),
        the final hidden state, and (each layer's updated RNNState).
    """
    num_layers = len(layer_states)
    new_states = []
    cell_states = []
    hidden_states = []
    state = lstm_input
    for i in range(num_layers):
        # view as (1, input_size)
        layer_h, layer_c = rnns[i](torch.unsqueeze(state,0), layer_states[i])
        new_states.append((layer_h, layer_c))

        layer_h = layer_h.squeeze()
        layer_c = layer_c.squeeze()

        state = layer_h
        if i < num_layers - 1:
            # In both Dynet and Pytorch
            # p stands for probability of an element to be zeroed. i.e. p=1 means switch off all activations.
            state = F.dropout(state, p=dropout_amount, training=rnns[i].training)

        cell_states.append(layer_c)
        hidden_states.append(layer_h)

    return (cell_states, hidden_states), state, new_states


def encode_sequence(sequence, rnns, embedder, dropout_amount=0.):
    """ Encodes a sequence given RNN cells and an embedding function.

    Inputs:
        seq (list of str): The sequence to encode.
        rnns (list of dy._RNNBuilder): The RNNs to use.
        emb_fn (dict str->dy.Expression): Function that embeds strings to
            word vectors.
        size (int): The size of the RNN.
        dropout_amount (float, optional): The amount of dropout to apply.

    Returns:
        (list of dy.Expression, list of dy.Expression), list of dy.Expression,
        where the first pair is the (final cell memories, final cell states) of
        all layers, and the second list is a list of the final layer's cell
        state for all tokens in the sequence.
    """

    batch_size = 1
    layer_states = []
    for rnn in rnns:
        hidden_size = rnn.weight_hh.size()[1]
        
        # h_0 of shape (batch, hidden_size)
        # c_0 of shape (batch, hidden_size)
        if rnn.weight_hh.is_cuda:
            h_0 = torch.cuda.FloatTensor(batch_size,hidden_size).fill_(0)
            c_0 = torch.cuda.FloatTensor(batch_size,hidden_size).fill_(0)
        else:
            h_0 = torch.zeros(batch_size,hidden_size)
            c_0 = torch.zeros(batch_size,hidden_size)

        layer_states.append((h_0, c_0))

    outputs = []
    for token in sequence:
        rnn_input = embedder(token)
        (cell_states, hidden_states), output, layer_states = forward_one_multilayer(rnns,rnn_input,layer_states,dropout_amount)

        outputs.append(output)

    return (cell_states, hidden_states), outputs

def create_multilayer_lstm_params(num_layers, in_size, state_size, name=""):
    """ Adds a multilayer LSTM to the model parameters.

    Inputs:
        num_layers (int): Number of layers to create.
        in_size (int): The input size to the first layer.
        state_size (int): The size of the states.
        model (dy.ParameterCollection): The parameter collection for the model.
        name (str, optional): The name of the multilayer LSTM.
    """
    lstm_layers = []
    for i in range(num_layers):
        layer_name = name + "-" + str(i)
        print("LSTM " + layer_name + ": " + str(in_size) + " x " + str(state_size) + "; default Dynet initialization of hidden weights")
        lstm_layer = torch.nn.LSTMCell(input_size=int(in_size), hidden_size=int(state_size), bias=True)
        lstm_layers.append(lstm_layer)
        in_size = state_size
    return torch.nn.ModuleList(lstm_layers)

def add_params(size, name=""):
    """ Adds parameters to the model.

    Inputs:
        model (dy.ParameterCollection): The parameter collection for the model.
        size (tuple of int): The size to create.
        name (str, optional): The name of the parameters.
    """
    if len(size) == 1:
        print("vector " + name + ": " + str(size[0]) + "; uniform in [-0.1, 0.1]")
    else:
        print("matrix " + name + ": " + str(size[0]) + " x " + str(size[1]) + "; uniform in [-0.1, 0.1]")

    size_int = tuple([int(ss) for ss in size])
    use_gauss = False
    if use_gauss:
        return torch.nn.Parameter(torch.empty(size_int).normal_(0, 0.01))
    else:
        return torch.nn.Parameter(torch.empty(size_int).uniform_(-0.1, 0.1))

def mulrel(self, src_emb, tar_emb, src_mask, tar_mask, mul_score_w, \
    mul_emb_w=None, src_none_node=None, tar_none_node=None, softmax_src=True, \
        tar_sum=True, use_tar_node=False):
        '''
        src_emb: batchsize, src_num, dim
        tar_emb: batchsize, tar_num, dim
        mul_score_w, mul_emb_w: rel_num, dim
        src_none_node, tar_none_node: dim
        softmax_src: do softmax in src index
        tar_sum: whether sum in tar index
        use_tar_node: whether add tar_none_node into src
        '''
        batchsize = src_emb.size(0)
        if src_none_node is not None:
            none_node_mask = torch.ones(batchsize, 1).cuda()
            src_mask_ = torch.cat([src_mask, none_node_mask], dim=1)
            src_emb_ = torch.cat([src_emb, src_none_node.view(1, 1, -1).expand(batchsize, 1, src_emb.size(2))], dim=1)
        else:
            src_mask_ = src_mask
            src_emb_ = src_emb
        if tar_none_node is not None:
            none_node_mask = torch.ones(batchsize, 1).cuda()
            tar_mask_ = torch.cat([tar_mask, none_node_mask], dim=1)
            tar_emb_ = torch.cat([tar_emb, tar_none_node.view(1, 1, -1).expand(batchsize, 1, tar_emb.size(2))], dim=1)
        else:
            tar_mask_ = tar_mask
            tar_emb_ = tar_emb

        scores = src_emb_.unsqueeze(0) * mul_score_w.view(mul_score_w.size(0), 1, 1, -1) # rel_num, batch, src_num, dim
        # rel_num, batch, src_num, tar_num
        src_tar_score = torch.sum(scores.unsqueeze(3) * tar_emb_.view(1, tar_emb_.size(0), 1, tar_emb_.size(1), -1), dim=-1)
        
        src_tar_mask = src_mask_.unsqueeze(2) * tar_mask_.unsqueeze(1)
        src_tar_score = src_tar_score.add((1-src_tar_mask).mul(-1e10).unsqueeze(0)).mul(1 / np.sqrt(src_emb_.size(2)))
        
        if softmax_src:
            src_tar_weight = F.softmax(src_tar_score, dim=2) # rel_num, batch, src_num, tar_num
        else:
            src_tar_weight = F.softmax(src_tar_score, dim=3)

        if mul_emb_w is not None:
            rel_tar_emb = tar_emb_.unsqueeze(0) * mul_emb_w.view(mul_emb_w.size(0), 1, 1, -1) # rel, batch, tar, dim
        else:
            rel_tar_emb = tar_emb_.unsqueeze(0)
        
        if not use_tar_node:
            rel_tar_emb = rel_tar_emb[:, :, :-1, :]
            src_tar_weight = src_tar_weight[:, :, :, :-1]

        if tar_sum:
            # return : batch, src, dim
            if mul_emb_w is not None:
                mul_emb = torch.sum(torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=3), dim=0).mul(1. /mul_emb_w.size(0))
            else:
                mul_emb = torch.sum(torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=3), dim=0)
            
            if src_none_node is not None:
                mul_emb = mul_emb[:, :-1, :]
        else:
            # return: batch, src, tar, dim
            if mul_emb_w is not None:
                mul_emb = torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=0).mul(1. /mul_emb_w.size(0))
            else:
                mul_emb = torch.sum(src_tar_weight.unsqueeze(4) * rel_tar_emb.unsqueeze(2), dim=0)
            if src_none_node is not None:
                mul_emb = mul_emb[:, :-1, :, :]
        return mul_emb
