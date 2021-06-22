import torch
from .message_passing import MessagePassing
from torch.nn.init import xavier_normal_, uniform_
from torch.nn import Parameter
def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def get_param(shape):
	return torch.nn.Parameter(torch.empty(*shape).uniform_(-0.1, 0.1))
	# param = Parameter(torch.Tensor(*shape));
	# uniform_(param.data)
	# return param
def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)
def conj(a):
	a[..., 1] = -a[..., 1]
	return a

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, bias=None, dropout=0.1, opn='corr'):
		super(self.__class__, self).__init__()

		self.bias = bias
		self.dropout = dropout
		self.opn = opn
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_rels 	= num_rels
		self.act 		= act
		self.device		= None

		# self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		# self.w_rel 		= get_param((in_channels, out_channels))
		# self.loop_rel   = get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.dropout)
		# self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.bias:
			self.w_bias = Parameter(torch.zeros(out_channels))
	def forward(self, x, forward_edge_index, backward_edge_index, forward_edge_type, backward_edge_type, rel_embed):
		'''
		forward_edge_index, backward_edge_index: [(sub_node, obj_node), ...]
		forward_edge_type, backward_edge_type: [edge_type_index]
		'''
		if self.device is None:
			self.device = forward_edge_index.device

		if forward_edge_index.size(0) == 0:
			# print('get zeros edges')
			return torch.zeros_like(x).cuda(), rel_embed

		# print('forward_edge_index', forward_edge_index)
		# num_edges = forward_edge_index.size(1)

		# num_ent   = x.size(0)

		self.in_index, self.out_index = forward_edge_index, backward_edge_index
		self.in_type,  self.out_type  = forward_edge_type, backward_edge_type

		in_res	= self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=None, mode='in')
		out_res	= self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_norm=None, mode='out')
		out	= self.drop(in_res)*(1/2) + self.drop(out_res)*(1/2)
		if self.bias:
			out = out + self.w_bias
		# out = self.bn(out)

		# return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted
		return self.act(out), rel_embed		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		# print(ent_embed.size())
		# print(rel_embed.size())
		if   self.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError
		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		# print(x_j
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

if __name__ == "__main__":
	gcn = CompGCNConv(2, 2, 3, act=torch.tanh, bias=None, dropout=0.1, opn='corr')
	x = get_param((6, 2))
	print(x)
	forward_edge_index = torch.tensor([(2,3), (1,3),(5,4)]).t()
	# print('forward_edge_index', forward_edge_index.size())
	backward_edge_index = torch.tensor([(3,2), (3,1),(4,5)]).t()
	forward_edge_type = torch.tensor([0,1,2])
	backward_edge_type = torch.tensor([3,4,5])
	rel_embed = get_param((6, 2))
	print(rel_embed)
	a, r = gcn(x, forward_edge_index, backward_edge_index, forward_edge_type, backward_edge_type, rel_embed)
	print(a.size())
	print(a)
	print(r.size())
	print(r)