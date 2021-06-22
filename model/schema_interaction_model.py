""" Class for the Sequence to sequence model for ATIS."""
import os
import torch
import torch.nn.functional as F
from . import torch_utils

import data_util.snippets as snippet_handler
import data_util.sql_util
import data_util.vocabulary as vocab
from data_util.vocabulary import EOS_TOK, UNK_TOK 
import data_util.tokenizers
from apex import amp
from .token_predictor import construct_token_predictor
from .attention import Attention
from .mulrel import MulRel
from .dst_model import DSTModel
from .model import ATISModel, encode_snippets_with_states, get_token_indices
from data_util.utterance import ANON_INPUT_KEY

from .encoder import Encoder
from .decoder import SequencePredictorWithSchema
from .compgcn_conv import CompGCNConv

from . import utils_bert

import data_util.atis_batch


LIMITED_INTERACTIONS = {"raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
                        "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
                        "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1}

END_OF_INTERACTION = {"quit", "exit", "done"}

# COLUMN_LABELS = ['desc', 'intersect', 'avg', 'not', '(', 'order_by', '_EOS', 'union', 'min', 'having', 'and', '>',
#              'where', 'like', ')', 'limit_value', 'in', 'value', 'select', 'except', 'count', 'max', 'group_by', '-',
#              'asc', 'distinct', '!=', ',', '<', 'or', 'between', '+', 'sum', '=', '_UNK']

class SchemaInteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self,
                 params,
                 input_vocabulary,
                 output_vocabulary,
                 output_vocabulary_schema,
                 anonymizer):
        ATISModel.__init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            output_vocabulary_schema,
            anonymizer)
        self.column_labels = output_vocabulary.raw_vocab.id_to_token
        self.column_labels_to_id = {}
        for lab in self.column_labels:
            self.column_labels_to_id[lab] = len(self.column_labels_to_id)

        if self.params.use_schema_encoder:
            schema_encoder_num_layer = 1
            schema_encoder_input_size = params.input_embedding_size
            schema_encoder_state_size = params.encoder_state_size
            if params.use_bert:
                schema_encoder_input_size = self.bert_config.hidden_size

            self.schema_encoder = Encoder(schema_encoder_num_layer, schema_encoder_input_size, schema_encoder_state_size)

        if self.params.gcn_edge_type == 'edge1':
            edge_rel = 2
        elif self.params.gcn_edge_type == 'edge2':
            edge_rel = 4
        else:
            raise 'edge type not implement'
        self.schema_cgcn = CompGCNConv(self.schema_attention_key_size, self.schema_attention_key_size, edge_rel, act=torch.tanh, bias=True, dropout=self.dropout, opn='mult')
        self.edge_rel_emb = torch_utils.add_params((edge_rel, self.schema_attention_key_size), "edge_rel-embedding")

        if self.params.column_label_type == 'binary':
            column_label_num = 2
        elif self.params.column_label_type == 'sql':
            column_label_num = len(self.column_labels) + 1
        elif self.params.column_label_type == 'semql':
            pass
        elif self.params.column_label_type == 'cross':
            self.column_none_label_emb = torch_utils.add_params(tuple([self.schema_attention_key_size]), "column-none-label-embedding")
            self.key_none_label_emb = torch_utils.add_params(tuple([self.schema_attention_key_size]), "key-none-label-embedding")
            column_label_num = len(self.column_labels)
        else:
            raise 'column label type not implement'

        self.output_transform_ = torch_utils.add_params((self.schema_attention_key_size, self.schema_attention_key_size), "previous-state-tranform")
        self.output_transform = lambda x: F.dropout(F.tanh(torch.mm(x.unsqueeze(0), self.output_transform_)).squeeze(0), p=self.dropout, training=self.training)
        if self.params.dst_loss_weight > 0:
            self.sql_dst = DSTModel(self.schema_attention_key_size, token_to_id=self.column_labels_to_id, dst_type='sql')
            self.key_dst = DSTModel(self.schema_attention_key_size, dst_type='keywords')
            self.sql_transform_ = torch_utils.add_params((self.schema_attention_key_size, self.schema_attention_key_size), "sql-label-state-tranform")
            self.sql_transform = lambda x: F.dropout(F.tanh(torch.mm(x.unsqueeze(0), self.sql_transform_)).squeeze(0), p=self.dropout, training=self.training)
            self.key_transform_ = torch_utils.add_params((self.schema_attention_key_size, self.schema_attention_key_size), "key-label-state-tranform")
            self.key_transform = lambda x: F.dropout(F.tanh(torch.mm(x.unsqueeze(0), self.key_transform_)).squeeze(0), p=self.dropout, training=self.training)

        # utterance level pointer attention
        if self.params.use_utterance_mulrel:
            utterance_rel_num = 3
            mulrel_size = self.params.encoder_state_size
            self.uttrerance_mulrel = MulRel(utterance_rel_num, mulrel_size, use_emb_w=False, \
                src_none_node=False, tar_none_node=False, softmax_src=False, \
            tar_sum=True, use_tar_node=False, diagonal_matrix=True, use_normalize=False, score_use_normalize=False, dropout_mount=self.dropout)
            self.utterance_mulrel_encoder = Encoder(1, mulrel_size*2, mulrel_size)

        if self.params.use_utterance_schema_mulrel:
            utt2sche_rel_num = 3
            mulrel_size = self.params.encoder_state_size
            self.utterance_schema_mulrel = MulRel(utt2sche_rel_num, mulrel_size, use_emb_w=True, \
                src_none_node=False, tar_none_node=False, softmax_src=False, \
            tar_sum=True, use_tar_node=True, diagonal_matrix=True, use_normalize=False, score_use_normalize=False, dropout_mount=self.dropout)
            self.utterance_schema_mulrel_encoder = Encoder(1, mulrel_size*2, mulrel_size)
        # update schema
        if self.params.use_query_utterance_schema_mulrel:
            queryutt2schema_rel_num=5
            mulrel_size = self.params.encoder_state_size
            self.query_update_schema_mulrel = MulRel(queryutt2schema_rel_num, mulrel_size, use_emb_w=True, \
                src_none_node=False, tar_none_node=False, softmax_src=False, \
            tar_sum=True, use_tar_node=False, diagonal_matrix=True)
            self.utterance_update_schema_mulrel = MulRel(queryutt2schema_rel_num, mulrel_size, use_emb_w=True, \
                src_none_node=False, tar_none_node=False, softmax_src=False, \
            tar_sum=True, use_tar_node=False, diagonal_matrix=True)
            if self.params.use_query_utterance_schema_mulrel_mlp:
                self.query_utterance_schema_mulrel_mlp = torch_utils.add_params(tuple([params.encoder_state_size * 3, params.encoder_state_size]))
        else:
            if self.params.use_query_update_schema_mulrel:
                query2schema_rel_num=3
                mulrel_size = self.params.encoder_state_size
                self.query_update_schema_mulrel = MulRel(query2schema_rel_num, mulrel_size, use_emb_w=True, \
                    src_none_node=False, tar_none_node=False, softmax_src=False, \
                tar_sum=True, use_tar_node=True, diagonal_matrix=True, use_normalize=False, score_use_normalize=False, dropout_mount=self.dropout)
                if self.params.use_query_update_schema_mulrel_mlp:
                    self.query_update_schema_mulrel_mlp = torch_utils.add_params(tuple([params.encoder_state_size * 2, params.encoder_state_size]))
                else:
                    self.query_update_schema_mulrel_mlp = None
            if self.params.use_utterance_update_schema_mulrel:
                utterance2schema_rel_num=5
                mulrel_size = self.params.encoder_state_size
                self.utterance_update_schema_mulrel = MulRel(utterance2schema_rel_num, mulrel_size, use_emb_w=True, \
                    src_none_node=False, tar_none_node=False, softmax_src=False, \
                tar_sum=True, use_tar_node=False, diagonal_matrix=True)
                if self.params.use_utterance_update_schema_mulrel_mlp:
                    self.utterance_update_schema_mulrel_mlp = torch_utils.add_params(tuple([params.encoder_state_size * 2, params.encoder_state_size]))
                else:
                    self.utterance_update_schema_mulrel_mlp = None

        # utterance level attention
        if self.params.use_utterance_attention:
            self.utterance_attention_module = Attention(self.params.encoder_state_size, self.params.encoder_state_size, self.params.encoder_state_size)

        # input_hidden_states: self.utterance_attention_key_size x len(input)
        if params.use_encoder_attention:
            self.utterance2schema_attention_module = Attention(self.schema_attention_key_size, self.utterance_attention_key_size, self.utterance_attention_key_size)
            self.schema2utterance_attention_module = Attention(self.utterance_attention_key_size, self.schema_attention_key_size, self.schema_attention_key_size)

            new_attention_key_size = self.schema_attention_key_size + self.utterance_attention_key_size
            self.schema_attention_key_size = new_attention_key_size
            self.utterance_attention_key_size = new_attention_key_size

            if self.params.use_schema_encoder_2:
                self.schema_encoder_2 = Encoder(schema_encoder_num_layer, self.schema_attention_key_size, self.schema_attention_key_size)
                self.utterance_encoder_2 = Encoder(params.encoder_num_layers, self.utterance_attention_key_size, self.utterance_attention_key_size)

        self.token_predictor = construct_token_predictor(params,
                                                         output_vocabulary,
                                                         self.utterance_attention_key_size,
                                                         self.schema_attention_key_size,
                                                         self.final_snippet_size,
                                                         anonymizer)

        # Use schema_attention in decoder
        if params.use_schema_attention and params.use_query_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size + params.encoder_state_size
        elif params.use_schema_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size
        else:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size

        self.decoder = SequencePredictorWithSchema(params, decoder_input_size, self.output_embedder, self.column_name_token_embedder, self.token_predictor)
    def generate_column_appear_label_sql_cross(self, sql_sequence, schema):
        res = [['None'] for _ in range(len(schema.column_names_surface_form))]
        res_keywords = [[100000] for _ in range(len(self.column_labels))]
        last_labels = []
        for cur_token in sql_sequence:
            if cur_token in self.column_labels:
                last_labels.append(cur_token)
            if cur_token in schema.column_names_surface_form_to_id:
                cur_token_id = schema.column_names_surface_form_to_id[cur_token]
                if 'None' in res[cur_token_id]:
                    res[cur_token_id].remove('None')
                res[cur_token_id].extend(last_labels)
                last_labels = []
        for idx, labels in enumerate(res):
            if 'None' not in labels:
                for cur_label in labels:
                    if 100000 in res_keywords[self.column_labels_to_id[cur_label]]:
                        res_keywords[self.column_labels_to_id[cur_label]].remove(100000)
                    res_keywords[self.column_labels_to_id[cur_label]].extend([idx])
        return res, res_keywords

    def predict_turn(self,
                     utterance_final_state,
                     input_hidden_states,
                     schema_states,
                     vocabulary_emb,
                     max_generation_length,
                     previous_hidden_states=None,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     previous_queries=None,
                     previous_query_states=None,
                     input_schema=None,
                     feed_gold_tokens=False,
                     training=False):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        fed_sequence = []
        loss = None
        token_accuracy = 0.
        # update utterance
        if self.params.use_utterance_mulrel:
            if self.params.utterance_mulrel_series:
                final_mulrel_states, input_hidden_states = self.mulrel_utterance_series(input_hidden_states, previous_hidden_states)
            else:
                final_mulrel_states, input_hidden_states = self.mulrel_utterance_parallel(input_hidden_states, previous_hidden_states)
                if final_mulrel_states is not None:
                    utterance_final_state = final_mulrel_states

        if self.params.use_utterance_schema_mulrel:
            if self.params.use_utterance_after_schema:
                utterance_final_state, input_hidden_states = self.mulrel_utterance_schema(input_hidden_states, schema_states, previous_hidden_states)
            else:
                utterance_final_state, input_hidden_states = self.mulrel_utterance_schema(input_hidden_states, schema_states)
        # update schema
        schema_after = []
        if self.params.use_query_utterance_schema_mulrel:
            if len(previous_query_states) > 0:
                schema_after = self.mulrel_query_utterance_schema(schema_states, previous_hidden_states[-1], previous_query_states[-1])
            else:
                schema_after = self.mulrel_query_utterance_schema(schema_states, previous_hidden_states[-1])
            input_schema.set_column_name_embeddings(schema_after)
        else:
            if self.params.use_query_update_schema_mulrel:
                if len(previous_query_states) > 0:
                    schema_after = self.mulrel_update_schema(schema_states, previous_query_states[-1], self.query_update_schema_mulrel, self.query_update_schema_mulrel_mlp)
                else:
                    schema_after = schema_states
                input_schema.set_column_name_embeddings(schema_after)
            elif self.params.use_utterance_update_schema_mulrel:
                schema_after = self.mulrel_update_schema(schema_states, previous_hidden_states[-1], self.utterance_update_schema_mulrel, self.utterance_update_schema_mulrel_mlp)
                input_schema.set_column_name_embeddings(schema_after)
            else:
                schema_after = schema_states

        # print(utterance_final_state[0][-1].size())
        previous_schema_memory = [0.] * input_schema.num_col
        if len(previous_queries) > 0:
            for col in input_schema.column_names_surface_form_to_id:
                if col in previous_queries[-1]:
                    previous_schema_memory[input_schema.column_names_surface_form_to_id[col]] = 1.
        select_schema_memory = [0.] * input_schema.num_col

        if feed_gold_tokens:
            decoder_results = self.decoder(utterance_final_state,
                                        input_hidden_states,
                                        schema_after,
                                        vocabulary_emb,
                                        max_generation_length,
                                        gold_sequence=gold_query,
                                        input_sequence=input_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        previous_schema_memory=previous_schema_memory,
                                        select_schema_memory=select_schema_memory,
                                        input_schema=input_schema,
                                        snippets=snippets,
                                        dropout_amount=self.dropout)

            all_scores = []
            all_alignments = []
            for prediction in decoder_results.predictions:
                scores = F.softmax(prediction.scores, dim=0)
                alignments = prediction.aligned_tokens
                if self.params.use_previous_query and self.params.use_copy_switch and len(previous_queries) > 0:
                    query_scores = F.softmax(prediction.query_scores, dim=0)
                    copy_switch = prediction.copy_switch
                    scores = torch.cat([scores * (1 - copy_switch), query_scores * copy_switch], dim=0)
                    alignments = alignments + prediction.query_tokens
                all_scores.append(scores)
                all_alignments.append(alignments)
            # Compute the loss
            gold_sequence = gold_query
            if self.params.new_loss:
                loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices) / float(len(gold_query))
            else:
                loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices)

            if self.params.dst_loss_weight > 0:
                column_label, key_label = self.generate_column_appear_label_sql_cross(gold_query, input_schema)
                column_score = self.sql_dst(self.sql_transform(utterance_final_state[0][-1]), torch.stack(schema_after), vocabulary_emb)
                key_score = self.key_dst(self.key_transform(utterance_final_state[0][-1]), vocabulary_emb, torch.stack(schema_after))
                sql_loss = self.sql_dst.bce_loss(column_label)
                key_loss = self.key_dst.bce_loss(key_label)
                loss = loss + self.params.dst_loss_weight * (sql_loss + key_loss)

            if not training:
                if self.params.get_seq_from_scores_flatten:
                    predicted_sequence = torch_utils.get_seq_from_scores_flatten(all_scores, all_alignments)
                else:
                    predicted_sequence = torch_utils.get_seq_from_scores(all_scores, all_alignments)
                token_accuracy = torch_utils.per_token_accuracy(gold_sequence, predicted_sequence)
            fed_sequence = gold_sequence
        else:
            decoder_results = self.decoder(utterance_final_state,
                                        input_hidden_states,
                                        schema_after,
                                        vocabulary_emb,
                                        max_generation_length,
                                        input_sequence=input_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        previous_schema_memory=previous_schema_memory,
                                        select_schema_memory=select_schema_memory,
                                        input_schema=input_schema,
                                        snippets=snippets,
                                        dropout_amount=self.dropout)
            predicted_sequence = decoder_results.sequence
            fed_sequence = predicted_sequence
        # print('predicted_sequence', predicted_sequence)
        decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (predicted_sequence,
                loss,
                token_accuracy,
                decoder_states,
                decoder_results)

    def encode_schema_bow_simple(self, input_schema):
        schema_states = []
        for column_name in input_schema.column_names_embedder_input:
            schema_states.append(input_schema.column_name_embedder_bow(column_name, surface_form=False, column_name_token_embedder=self.column_name_token_embedder))
        input_schema.set_column_name_embeddings(schema_states)
        return schema_states

    def encode_schema_self_attention(self, schema_states):
        schema_self_attention = self.schema2schema_attention_module(torch.stack(schema_states,dim=0), schema_states).vector
        if schema_self_attention.dim() == 1:
            schema_self_attention = schema_self_attention.unsqueeze(1)
        residual_schema_states = list(torch.split(schema_self_attention, split_size_or_sections=1, dim=1))
        residual_schema_states = [schema_state.squeeze() for schema_state in residual_schema_states]

        new_schema_states = [schema_state+residual_schema_state for schema_state, residual_schema_state in zip(schema_states, residual_schema_states)]

        return new_schema_states
    def encode_schema_with_cgcn(self, schema_states, input_schema):
        schema_states_input = torch.stack(schema_states, dim=0)
        fw_edge_idx = torch.tensor(input_schema.fw_edge_index).cuda().t()
        bw_edge_idx = torch.tensor(input_schema.bw_edge_index).cuda().t()
        fw_edge_type = torch.tensor(input_schema.fw_edge_type).cuda()
        bw_edge_type = torch.tensor(input_schema.bw_edge_type).cuda()
        x, r = self.schema_cgcn(schema_states_input, fw_edge_idx, bw_edge_idx, fw_edge_type, bw_edge_type, self.edge_rel_emb)
        schema_states_input = schema_states_input + x
        residual_schema_states = list(torch.split(schema_states_input, split_size_or_sections=1, dim=0))
        residual_schema_states = [schema_state.squeeze() for schema_state in residual_schema_states]
        return residual_schema_states

    def encode_schema(self, input_schema, dropout=False):
      schema_states = []
      for column_name_embedder_input in input_schema.column_names_embedder_input:
        tokens = column_name_embedder_input.split()

        if dropout:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder, dropout_amount=self.dropout)
        else:
          final_schema_state_one, schema_states_one = self.schema_encoder(tokens, self.column_name_token_embedder)
        schema_states.append(final_schema_state_one[1][-1])

      input_schema.set_column_name_embeddings(schema_states)

      # self-attention over schema_states
      if self.params.use_schema_self_attention:
        schema_states = self.encode_schema_self_attention(schema_states)

      return schema_states

    def get_bert_encoding(self, input_sequence, input_schema, discourse_state, dropout):
        utterance_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, input_schema, bert_input_version=self.params.bert_input_version, num_out_layers_n=1, num_out_layers_h=1)

        if self.params.discourse_level_lstm:
            utterance_token_embedder = lambda x: torch.cat([x, discourse_state], dim=0)
        else:
            utterance_token_embedder = lambda x: x

        if dropout:
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder,
                dropout_amount=self.dropout)
        else:
            final_utterance_state, utterance_states = self.utterance_encoder(
                utterance_states,
                utterance_token_embedder)

        schema_states = []
        for schema_token_states1 in schema_token_states:
            if dropout:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            else:
                final_schema_state_one, schema_states_one = self.schema_encoder(schema_token_states1, lambda x: x)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            schema_states.append(final_schema_state_one[1][-1])
        del schema_token_states
        input_schema.set_column_name_embeddings(schema_states)

        return final_utterance_state, utterance_states, schema_states

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_query_token_embedding_previous(self, output_token, input_schema, vocabulary_emb):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = vocabulary_emb[self.output_embedder.vocab_token_lookup(output_token)]
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = vocabulary_emb[self.output_embedder.vocab_token_lookup(output_token)]
        return output_token_embedding

    def get_utterance_attention(self, final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep):
        # self-attention between utterance_states
        final_utterance_states_c.append(final_utterance_state[0][0])
        final_utterance_states_h.append(final_utterance_state[1][0])
        final_utterance_states_c = final_utterance_states_c[-num_utterances_to_keep:]
        final_utterance_states_h = final_utterance_states_h[-num_utterances_to_keep:]

        attention_result = self.utterance_attention_module(final_utterance_states_c[-1], final_utterance_states_c)
        final_utterance_state_attention_c = final_utterance_states_c[-1] + attention_result.vector.squeeze()

        attention_result = self.utterance_attention_module(final_utterance_states_h[-1], final_utterance_states_h)
        final_utterance_state_attention_h = final_utterance_states_h[-1] + attention_result.vector.squeeze()

        final_utterance_state = ([final_utterance_state_attention_c],[final_utterance_state_attention_h])

        return final_utterance_states_c, final_utterance_states_h, final_utterance_state

    def get_previous_queries(self, previous_queries, previous_query_states, previous_query, input_schema):
        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)
        _, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout)
        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        return previous_queries, previous_query_states
    def get_previous_queries_previous(self, previous_queries, previous_query_states, previous_query, input_schema, vocabulary_emb):
        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        query_token_embedder = lambda query_token: self.get_query_token_embedding_previous(query_token, input_schema, vocabulary_emb)
        _, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout)
        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        return previous_queries, previous_query_states

    def train_step(self, interaction, max_generation_length, snippet_alignment_probability=1.):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """
        # assert self.params.discourse_level_lstm

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []
        previous_hidden_states = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        if self.params.gcn_edge_type == 'edge1':
            input_schema.num_edge = input_schema.set_schema_graph()
        elif self.params.gcn_edge_type == 'edge2':
            input_schema.num_edge = input_schema.set_schema_graph_2()
        else:
            pass
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)
        res_output_emb = []
        for utterance_index, utterance in enumerate(interaction.gold_utterances()):
            if interaction.identifier in LIMITED_INTERACTIONS and utterance_index > LIMITED_INTERACTIONS[interaction.identifier]:
                break
            input_sequence = utterance.input_sequence()
            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                else:
                    pass

            # Get the gold query: reconstruct if the alignment probability is less than one
            if snippet_alignment_probability < 1.:
                gold_query = sql_util.add_snippets_to_query(
                    available_snippets,
                    utterance.contained_entities(),
                    utterance.anonymized_gold_query(),
                    prob_align=snippet_alignment_probability) + [vocab.EOS_TOK]
            else:
                gold_query = utterance.gold_query()

            # Encode the utterance, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))
            flat_sequence = []
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        pass
                    else:
                        previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            if self.params.column_label_type == 'binary':
                column_label = input_schema.generate_column_appear_label_binary(previous_query)
            elif self.params.column_label_type == 'sql':
                column_label = input_schema.generate_column_appear_label_sql_sequential(previous_query)
            elif self.params.column_label_type == 'cross':
                column_label, key_label = self.generate_column_appear_label_sql_cross(previous_query, input_schema)
            elif self.params.column_label_type == 'semql':
                pass
            else:
                pass
            # schema labels update
            new_states = []
            for cur_schema_states, cur_column_label in zip(schema_states, column_label):
                for cur_cur_label in cur_column_label:
                    if cur_cur_label != 'None':
                        cur_schema_states = cur_schema_states + self.output_embedder(cur_cur_label)
                    else:
                        cur_schema_states = cur_schema_states + self.column_none_label_emb
                new_states.append(cur_schema_states)
            schema_states = new_states
            schema_states = self.encode_schema_with_cgcn(schema_states, input_schema)
            # keywords update
            output_emd_matrix = self.output_embedder.token_embedding_matrix.weight
            res_output_emb = []
            for idx, key_labels in enumerate(key_label):
                cur_emb_ = output_emd_matrix[idx]
                for idx_col in key_labels:
                    if idx_col != 100000:
                        cur_emb_ = cur_emb_ + input_schema.column_name_embeddings[idx_col]
                    else:
                        cur_emb_ = cur_emb_ + self.key_none_label_emb
                res_output_emb.append(self.output_transform(cur_emb_))
            res_output_emb = torch.stack(res_output_emb)
            input_schema.set_column_name_embeddings(schema_states)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                    else:
                        pass

            if len(gold_query) <= max_generation_length and len(previous_query) <= max_generation_length:
                prediction = self.predict_turn(final_utterance_state,
                                               utterance_states,
                                               schema_states,
                                               res_output_emb,
                                               max_generation_length,
                                               previous_hidden_states=previous_hidden_states,
                                               gold_query=gold_query,
                                               snippets=snippets,
                                               input_sequence=flat_sequence,
                                               previous_queries=previous_queries,
                                               previous_query_states=previous_query_states,
                                               input_schema=input_schema,
                                               feed_gold_tokens=True,
                                               training=True)
                loss = prediction[1]
                # print('total_loss', loss)
                decoder_states = prediction[3]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

            torch.cuda.empty_cache()

        if losses:
            if self.params.new_loss:
                average_loss = torch.sum(torch.stack(losses)) / len(losses)
                normalized_loss = average_loss
            else:
                average_loss = torch.sum(torch.stack(losses)) / total_gold_tokens
                normalized_loss = average_loss
                if self.params.reweight_batch:
                    normalized_loss = len(losses) * average_loss / float(self.params.batch_size)

            if self.params.norm_loss:
                e = 1e-7
                # if self.params.use_utterance_mulrel:
                #     normalized_loss += self.uttrerance_mulrel.eu_loss(e)
                # if self.params.use_utterance_schema_mulrel:
                #     normalized_loss += self.utterance_schema_mulrel.eu_loss(e)
                # if self.params.use_query_utterance_schema_mulrel:
                #     normalized_loss += self.query_update_schema_mulrel.eu_loss(e)
                #     normalized_loss += self.utterance_update_schema_mulrel.eu_loss(e)
                # else:
                #     if self.params.use_query_update_schema_mulrel:
                #         normalized_loss += self.query_update_schema_mulrel.eu_loss(e)
                #     if self.params.use_utterance_update_schema_mulrel:
                #         normalized_loss += self.utterance_update_schema_mulrel.eu_loss(e)

                if self.params.use_utterance_mulrel:
                    normalized_loss += self.uttrerance_mulrel.min_cosine_loss(e)
                if self.params.use_utterance_schema_mulrel:
                    normalized_loss += self.utterance_schema_mulrel.min_cosine_loss(e)
                if self.params.use_query_utterance_schema_mulrel:
                    normalized_loss += self.query_update_schema_mulrel.min_cosine_loss(e)
                    normalized_loss += self.utterance_update_schema_mulrel.min_cosine_loss(e)
                else:
                    if self.params.use_query_update_schema_mulrel:
                        normalized_loss += self.query_update_schema_mulrel.min_cosine_loss(e)
                    if self.params.use_utterance_update_schema_mulrel:
                        normalized_loss += self.utterance_update_schema_mulrel.min_cosine_loss(e)

            # normalized_loss.backward()
            with amp.scale_loss(normalized_loss, [self.trainer, self.bert_trainer]) as scaled_loss:
                scaled_loss.backward()
            self.trainer.step()
            if self.params.fine_tune_bert:
                self.bert_trainer.step()
            self.zero_grad()
            loss_scalar = normalized_loss.item()
        else:
            loss_scalar = 0.

        return loss_scalar

    def predict_with_predicted_queries(self, interaction, max_generation_length, syntax_restrict=True):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm

        syntax_restrict=False

        predictions = []

        input_hidden_states = []
        previous_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        if self.params.gcn_edge_type == 'edge1':
            input_schema.num_edge = input_schema.set_schema_graph()
        elif self.params.gcn_edge_type == 'edge2':
            input_schema.num_edge = input_schema.set_schema_graph_2()
        else:
            pass
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        interaction.start_interaction()
        while not interaction.done():
            utterance = interaction.next_utterance()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            input_sequence = utterance.input_sequence()
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                else:
                    pass

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=False)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))
            flat_sequence = []
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)
            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets, input_schema)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        pass
                    else:
                        previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            if self.params.column_label_type == 'binary':
                column_label = input_schema.generate_column_appear_label_binary(previous_query)
            elif self.params.column_label_type == 'sql':
                column_label = input_schema.generate_column_appear_label_sql_sequential(previous_query)
            elif self.params.column_label_type == 'cross':
                column_label, key_label = self.generate_column_appear_label_sql_cross(previous_query, input_schema)
                # print('column_label', column_label)
                # print('key_label', key_label)
            elif self.params.column_label_type == 'semql':
                pass
            else:
                pass
            # schema labels update
            new_states = []
            for cur_schema_states, cur_column_label in zip(schema_states, column_label):
                for cur_cur_label in cur_column_label:
                    if cur_cur_label != 'None':
                        cur_schema_states = cur_schema_states + self.output_embedder(cur_cur_label)
                    else:
                        cur_schema_states = cur_schema_states + self.column_none_label_emb
                new_states.append(cur_schema_states)
            schema_states = new_states
            schema_states = self.encode_schema_with_cgcn(schema_states, input_schema)
            # keywords update
            output_emd_matrix = self.output_embedder.token_embedding_matrix.weight
            res_output_emb = []
            for idx, key_labels in enumerate(key_label):
                cur_emb_ = output_emd_matrix[idx]
                for idx_col in key_labels:
                    if idx_col != 100000:
                        cur_emb_ = cur_emb_ + input_schema.column_name_embeddings[idx_col]
                    else:
                        cur_emb_ = cur_emb_ + self.key_none_label_emb
                res_output_emb.append(self.output_transform(cur_emb_))
            res_output_emb = torch.stack(res_output_emb)
            input_schema.set_column_name_embeddings(schema_states)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                    else:
                        pass

            results = self.predict_turn(final_utterance_state,
                                        utterance_states,
                                        schema_states,
                                        res_output_emb,
                                        max_generation_length,
                                        previous_hidden_states=previous_hidden_states,
                                        input_sequence=flat_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        input_schema=input_schema,
                                        snippets=snippets)

            predicted_sequence = results[0]
            predictions.append(results)

            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)
            if EOS_TOK in anonymized_sequence:
                anonymized_sequence = anonymized_sequence[:-1] # Remove _EOS
            else:
                anonymized_sequence = ['select', '*', 'from', 't1']

            if not syntax_restrict:
                utterance.set_predicted_query(interaction.remove_snippets(predicted_sequence))
                if input_schema:
                    # on SParC
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=True)
                else:
                    # on ATIS
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=False)
            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(utterance, utterance.previous_query(), previous_snippets=utterance.snippets())

        return predictions


    def predict_with_gold_queries(self, interaction, max_generation_length, feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        # assert self.params.discourse_level_lstm

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        previous_hidden_states = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        if self.params.gcn_edge_type == 'edge1':
            input_schema.num_edge = input_schema.set_schema_graph()
        elif self.params.gcn_edge_type == 'edge2':
            input_schema.num_edge = input_schema.set_schema_graph_2()
        else:
            pass
        schema_states = []
        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)


        for utterance in interaction.gold_utterances():
            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                else:
                    pass
            # Encode the utterance, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence,
                    utterance_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))
            flat_sequence = []
            for utt in input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)
            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        pass
                    else:
                        previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)


            if self.params.column_label_type == 'binary':
                column_label = input_schema.generate_column_appear_label_binary(previous_query)
            elif self.params.column_label_type == 'sql':
                column_label = input_schema.generate_column_appear_label_sql_sequential(previous_query)
            elif self.params.column_label_type == 'cross':
                column_label, key_label = self.generate_column_appear_label_sql_cross(previous_query, input_schema)
            elif self.params.column_label_type == 'semql':
                pass
            else:
                pass
            # schema labels update
            new_states = []
            for cur_schema_states, cur_column_label in zip(schema_states, column_label):
                for cur_cur_label in cur_column_label:
                    if cur_cur_label != 'None':
                        cur_schema_states = cur_schema_states + self.output_embedder(cur_cur_label)
                    else:
                        cur_schema_states = cur_schema_states + self.column_none_label_emb
                new_states.append(cur_schema_states)
            schema_states = new_states
            schema_states = self.encode_schema_with_cgcn(schema_states, input_schema)
            # keywords update
            output_emd_matrix = self.output_embedder.token_embedding_matrix.weight
            res_output_emb = []
            for idx, key_labels in enumerate(key_label):
                cur_emb_ = output_emd_matrix[idx]
                for idx_col in key_labels:
                    if idx_col != 100000:
                        cur_emb_ = cur_emb_ + input_schema.column_name_embeddings[idx_col]
                    else:
                        cur_emb_ = cur_emb_ + self.key_none_label_emb
                res_output_emb.append(self.output_transform(cur_emb_))
            res_output_emb = torch.stack(res_output_emb)
            input_schema.set_column_name_embeddings(schema_states)
            if self.params.use_previous_query and len(previous_query) > 0:
                if self.params.use_previous_emb_for_previous_query:
                    pass
                else:
                    if self.params.use_current_emb_dynamic:
                        previous_queries, previous_query_states = self.get_previous_queries_previous(previous_queries, previous_query_states, previous_query, input_schema, res_output_emb)
                    else:
                        pass

            prediction = self.predict_turn(final_utterance_state,
                                           utterance_states,
                                           schema_states,
                                           res_output_emb,
                                           max_generation_length,
                                           previous_hidden_states=previous_hidden_states,
                                           gold_query=utterance.gold_query(),
                                           snippets=snippets,
                                           input_sequence=flat_sequence,
                                           previous_queries=previous_queries,
                                           previous_query_states=previous_query_states,
                                           input_schema=input_schema,
                                           feed_gold_tokens=feed_gold_query)

            decoder_states = prediction[3]
            predictions.append(prediction)
        return predictions

    def mulrel_utterance_series(self, input_hidden_states, previous_hidden_states):
        src_hidden_states = torch.stack(input_hidden_states, dim=0)
        cat_state_list = []
        hidden_state_num = len(previous_hidden_states)
        if hidden_state_num == 0:
            utterance_states = input_hidden_states
            previous_hidden_states.append(utterance_states)
            final_utterance_state = None
        else:
            tar_hidden_states = torch.stack(previous_hidden_states[-1], dim=0)
            src_mask = torch.ones(1, src_hidden_states.size(0)).cuda()
            tar_mask = torch.ones(1, tar_hidden_states.size(0)).cuda()
            mulrel_states = self.uttrerance_mulrel(src_hidden_states.unsqueeze(0), tar_hidden_states.unsqueeze(0), src_mask, tar_mask, dropout_amount=self.dropout).squeeze(0)
            cat_states = torch.cat([src_hidden_states, mulrel_states], dim=1)
            for i in range(len(input_hidden_states)):
                cat_state_list.append(cat_states[i])
            final_utterance_state, utterance_states = self.utterance_mulrel_encoder(cat_state_list, lambda x: x)
            previous_hidden_states.append(utterance_states)

        return final_utterance_state, utterance_states
    def mulrel_utterance_parallel(self, input_hidden_states, previous_hidden_states):
        src_hidden_states = torch.stack(input_hidden_states, dim=0)
        cat_state_list = []
        hidden_state_num = len(previous_hidden_states)
        if hidden_state_num == 0:
            utterance_states = input_hidden_states
            previous_hidden_states.append(utterance_states)
            final_utterance_state = None
        else:
            all_mulrel_states = []
            for i in range(hidden_state_num):
                tar_hidden_states = torch.stack(previous_hidden_states[i], dim=0)
                src_mask = torch.ones(1, src_hidden_states.size(0)).cuda()
                tar_mask = torch.ones(1, tar_hidden_states.size(0)).cuda()
                mulrel_states = self.uttrerance_mulrel(src_hidden_states.unsqueeze(0), tar_hidden_states.unsqueeze(0), src_mask, tar_mask, dropout_amount=self.dropout).squeeze(0)
                all_mulrel_states.append(mulrel_states)
            all_mulrel_states = torch.mean(torch.stack(all_mulrel_states, dim=0), dim=0)
            cat_states = torch.cat([src_hidden_states, all_mulrel_states], dim=1)
            for i in range(len(input_hidden_states)):
                cat_state_list.append(cat_states[i])
            final_utterance_state, utterance_states = self.utterance_mulrel_encoder(cat_state_list, lambda x: x)
            previous_hidden_states.append(input_hidden_states)
        return final_utterance_state, utterance_states
        
    def mulrel_utterance_schema(self, input_hidden_states, schema_states, previous_hidden_states=None):
        src_hidden_states = torch.stack(input_hidden_states, dim=0)
        tar_hidden_states = torch.stack(schema_states, dim=0)
        cat_state_list = []
        src_mask = torch.ones(1, src_hidden_states.size(0)).cuda()
        tar_mask = torch.ones(1, tar_hidden_states.size(0)).cuda()

        mulrel_states = self.utterance_schema_mulrel(src_hidden_states.unsqueeze(0), tar_hidden_states.unsqueeze(0), src_mask, tar_mask, dropout_amount=self.dropout).squeeze(0)
        cat_states = torch.cat([src_hidden_states, mulrel_states], dim=1)
        for i in range(len(input_hidden_states)):
            cat_state_list.append(cat_states[i])
        final_utterance_state, utterance_states = self.utterance_schema_mulrel_encoder(cat_state_list, lambda x: x)
        # previous_hidden_states.append(utterance_states)
        if previous_hidden_states is not None:
            previous_hidden_states[-1] = utterance_states
        return final_utterance_state, utterance_states
    def mulrel_original(self, src, tar, mulrel_ins):
        src_mask = torch.ones(1, src.size(0)).cuda()
        tar_mask = torch.ones(1, tar.size(0)).cuda()
        return mulrel_ins(src.unsqueeze(0), tar.unsqueeze(0), src_mask, tar_mask, dropout_amount=self.dropout).squeeze(0)

    def mulrel_query_utterance_schema(self, src, tar1, tar2=None):
        schema_after = []
        schema_after_1 = self.mulrel_original(torch.stack(src), torch.stack(tar1), self.query_update_schema_mulrel)
        if tar2 is not None:
            schema_after_2 = self.mulrel_original(torch.stack(src), torch.stack(tar2), self.utterance_update_schema_mulrel)
        else:
            schema_after_2 = torch.zeros(len(src), src[0].size(0)).cuda()
        if self.params.use_query_utterance_schema_mulrel_mlp:
            schema_after_ = torch.cat([torch.stack(src), schema_after_1], dim=1)
            schema_after_ = torch.cat([schema_after_, schema_after_2], dim=1)
            final_schema = torch.tanh(torch.mm(schema_after_, self.query_utterance_schema_mulrel_mlp))
            for i in range(len(src)):
                schema_after.append(final_schema[i])
        else:
            for i in range(len(src)):
                schema_after.append(src[i] + schema_after_1[i]+schema_after_2[i])
        return schema_after
    def mulrel_update_schema(self, src, tar, mulrel_ins, mlp=None):
        schema_after = []
        schema_after_1 = self.mulrel_original(torch.stack(src), torch.stack(tar), mulrel_ins)
        if mlp is not None:
            schema_after_ = torch.cat([torch.stack(src), schema_after_1], dim=1)
            final_schema = torch.tanh(torch.mm(schema_after_, mlp))
            for i in range(len(src)):
                schema_after.append(final_schema[i])
        else:
            for i in range(len(src)):
                schema_after.append(src[i] + schema_after_1[i])
        return schema_after