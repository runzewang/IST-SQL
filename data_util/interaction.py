""" Contains the class for an interaction in ATIS. """
import torch
from . import anonymization as anon
from . import sql_util
from .snippets import expand_snippets
from .utterance import Utterance, OUTPUT_KEY, ANON_INPUT_KEY

class Schema:
    def __init__(self, table_schema, simple=False):
        # self.column_labels = ['distinct', 'where', '!=', 'order_by', 'min', 'count', 'in', 'avg', 'between', 'and', 'except', 'group_by', '<',
        #      'select', '>', 'max', '=', 'like', 'asc', 'intersect', 'having', 'sum', 'union', 'desc']
        self.column_labels = ['desc', 'intersect', 'avg', 'not', '(', 'order_by', '_EOS', 'union', 'min', 'having', 'and', '>',
             'where', 'like', ')', 'limit_value', 'in', 'value', 'select', 'except', 'count', 'max', 'group_by', '-',
             'asc', 'distinct', '!=', ',', '<', 'or', 'between', '+', 'sum', '=', '_UNK']
        self.column_labels_to_id = {}
        for lab in self.column_labels:
            self.column_labels_to_id[lab] = len(self.column_labels_to_id) + 1 # 0 is for the none label
        self.column_label_nums = len(self.column_labels) + 1
        if simple:
            self.helper1(table_schema)
        else:
            self.helper2(table_schema)

    def helper1(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index.append(i)

        column_keep_index_2 = []
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = table_name.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index_2.append(i)

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name
            if i in column_keep_index_2:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        max_id_1 = max(v for k,v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface_form)

    def helper2(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        # print(table_schema.keys())
        self.foreign_key = table_schema['foreign_keys']

        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(table_name,column_name)
            else:
                column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index.append(i)
            else:
                print('same column name')
                print('table_schema', table_schema)

        start_i = len(self.column_names_surface_form_to_id)
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = '{}.*'.format(table_name.lower())
            self.column_names_surface_form.append(column_name_surface_form)
            self.column_names_surface_form_to_id[column_name_surface_form] = i + start_i

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        start_i = len(self.column_names_embedder_input_to_id)
        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            self.column_names_embedder_input.append(column_name_embedder_input)
            self.column_names_embedder_input_to_id[column_name_embedder_input] = i + start_i

        assert len(self.column_names_surface_form) == len(self.column_names_surface_form_to_id) == len(self.column_names_embedder_input) == len(self.column_names_embedder_input_to_id)

        max_id_1 = max(v for k,v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1
        # self.num_edge = self.set_schema_graph_2()
        self.num_col = len(self.column_names_surface_form)

    def __len__(self):
        return self.num_col

    def in_vocabulary(self, column_name, surface_form=False):
        if surface_form:
            return column_name in self.column_names_surface_form_to_id
        else:
            return column_name in self.column_names_embedder_input_to_id
    def generate_column_appear_label_binary(self, sql_sequence):
        res = []
        for cur_column in self.column_names_surface_form:
            if cur_column in sql_sequence:
                res.append([1])
            else:
                res.append([0])
        return res

    def generate_column_appear_label_sql(self, sql_sequence):
        res = [[0] for _ in range(len(self.column_names_surface_form))]
        res_label = [['None'] for _ in range(len(self.column_names_surface_form))]
        last_label = None
        for cur_token in sql_sequence:
            if cur_token in self.column_labels:
                last_label = cur_token
            if cur_token in self.column_names_surface_form_to_id:
                cur_token_id = self.column_names_surface_form_to_id[cur_token]

                if 0 in res[cur_token_id]:
                    res[cur_token_id].remove(0)

                if 'None' in res_label[cur_token_id]:
                    res_label[cur_token_id].remove('None')

                res[cur_token_id].append(self.column_labels_to_id[last_label])
                res_label[cur_token_id].append(last_label)
        return res
    def generate_column_appear_label_sql_sequential(self, sql_sequence):
        res = [[0] for _ in range(len(self.column_names_surface_form))]
        res_label = [['None'] for _ in range(len(self.column_names_surface_form))]
        last_labels = []
        for cur_token in sql_sequence:
            if cur_token in self.column_labels:
                last_labels.append(self.column_labels_to_id[cur_token])
            if cur_token in self.column_names_surface_form_to_id:
                cur_token_id = self.column_names_surface_form_to_id[cur_token]

                if 0 in res[cur_token_id]:
                    res[cur_token_id].remove(0)

                if 'None' in res_label[cur_token_id]:
                    res_label[cur_token_id].remove('None')

                res[cur_token_id].extend(last_labels)
                last_labels = []
        return res

    def generate_column_appear_label_sql_cross(self, sql_sequence):
        res = [[0] for _ in range(len(self.column_names_surface_form))]
        res_keywords = [[0] for _ in range(len(self.column_labels))]

        for cur_token in sql_sequence:
            if cur_token in self.column_labels:
                last_labels.append(self.column_labels_to_id[cur_token])
            if cur_token in self.column_names_surface_form_to_id:
                cur_token_id = self.column_names_surface_form_to_id[cur_token]

                if 0 in res[cur_token_id]:
                    res[cur_token_id].remove(0)
                res[cur_token_id].extend(last_labels)
                last_labels = []
        for idx, labels in enumerate(res):
            for cur_label in labels:
                if 0 in res_keywords[cur_label-1]:
                    res_keywords[cur_label-1].remove(0)
                res_keywords[cur_label-1].extend([idx])
        return res, res_keywords

    def column_name_embedder_bow(self, column_name, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
            column_name_embedder_input = self.column_names_embedder_input[column_name_id]
        else:
            column_name_embedder_input = column_name

        column_name_embeddings = [column_name_token_embedder(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)

    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col

    def column_name_embedder(self, column_name, surface_form=False):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
        else:
            column_name_id = self.column_names_embedder_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]
    def set_schema_graph(self):
        fw_edge_index = []
        fw_edge_type = []
        bw_edge_index = []
        bw_edge_type = []
        for keys in self.foreign_key:
            start, end = keys
            fw_edge_index.append((start, end)) #  -1 is because we do not set table.* at the begining.
            fw_edge_type.append(0)
            bw_edge_index.append((end, start))
            bw_edge_type.append(1)
        self.fw_edge_index = fw_edge_index
        self.fw_edge_type = fw_edge_type
        self.bw_edge_index = bw_edge_index
        self.bw_edge_type = bw_edge_type
        # print('self.column_names_surface_form', self.column_names_surface_form)
        # print('fw_edge_index', fw_edge_index)
        return max(bw_edge_type) + 1 if len(bw_edge_type) != 0 else 0

    def set_schema_graph_2(self):
        fw_edge_index = []
        fw_edge_type = []
        bw_edge_index = []
        bw_edge_type = []
        table_keys = []
        for keys in self.foreign_key:
            start, end = keys
            fw_edge_index.append((start, end))
            fw_edge_type.append(0)
            bw_edge_index.append((end, start))
            bw_edge_type.append(1)
            start_table_name = self.column_names_surface_form[start-1].split('.')[0]
            end_table_name = self.column_names_surface_form[end-1].split('.')[0]
            table_keys.append((start_table_name, end_table_name))
        for start_table_name, end_table_name in table_keys:
            for i, cur_column in enumerate(self.column_names_surface_form):
                table_name = cur_column.split('.')[0]
                if table_name == start_table_name:
                    for j, cur_column_2 in enumerate(self.column_names_surface_form):
                        table_name_2 = cur_column_2.split('.')[0]
                        if table_name_2 == end_table_name:
                            fw_edge_index.append((i,j))
                            fw_edge_type.append(2)
                            bw_edge_index.append((j,i))
                            bw_edge_type.append(3)
        self.fw_edge_index = fw_edge_index
        self.fw_edge_type = fw_edge_type
        self.bw_edge_index = bw_edge_index
        self.bw_edge_type = bw_edge_type
        # print('self.column_names_surface_form', self.column_names_surface_form)
        # print('fw_edge_index', fw_edge_index)
        return max(bw_edge_type) + 1 if len(bw_edge_type) != 0 else 0

class Interaction:
    """ ATIS interaction class.

    Attributes:
        utterances (list of Utterance): The utterances in the interaction.
        snippets (list of Snippet): The snippets that appear through the interaction.
        anon_tok_to_ent:
        identifier (str): Unique identifier for the interaction in the dataset.
    """
    def __init__(self,
                 utterances,
                 schema,
                 snippets,
                 anon_tok_to_ent,
                 identifier,
                 params):
        self.utterances = utterances
        self.schema = schema
        self.snippets = snippets
        self.anon_tok_to_ent = anon_tok_to_ent
        self.identifier = identifier

        # Ensure that each utterance's input and output sequences, when remapped
        # without anonymization or snippets, are the same as the original
        # version.
        for i, utterance in enumerate(self.utterances):
            deanon_input = self.deanonymize(utterance.input_seq_to_use,
                                            ANON_INPUT_KEY)
            assert deanon_input == utterance.original_input_seq, "Anonymized sequence [" \
                + " ".join(utterance.input_seq_to_use) + "] is not the same as [" \
                + " ".join(utterance.original_input_seq) + "] when deanonymized (is [" \
                + " ".join(deanon_input) + "] instead)"
            desnippet_gold = self.expand_snippets(utterance.gold_query_to_use)
            deanon_gold = self.deanonymize(desnippet_gold, OUTPUT_KEY)
            assert deanon_gold == utterance.original_gold_query, \
                "Anonymized and/or snippet'd query " \
                + " ".join(utterance.gold_query_to_use) + " is not the same as " \
                + " ".join(utterance.original_gold_query)

    def __str__(self):
        string = "Utterances:\n"
        for utterance in self.utterances:
            string += str(utterance) + "\n"
        string += "Anonymization dictionary:\n"
        for ent_tok, deanon in self.anon_tok_to_ent.items():
            string += ent_tok + "\t" + str(deanon) + "\n"

        return string

    def __len__(self):
        return len(self.utterances)

    def deanonymize(self, sequence, key):
        """ Deanonymizes a predicted query or an input utterance.

        Inputs:
            sequence (list of str): The sequence to deanonymize.
            key (str): The key in the anonymization table, e.g. NL or SQL.
        """
        return anon.deanonymize(sequence, self.anon_tok_to_ent, key)

    def expand_snippets(self, sequence):
        """ Expands snippets for a sequence.

        Inputs:
            sequence (list of str): A SQL query.

        """
        return expand_snippets(sequence, self.snippets)

    def input_seqs(self):
        in_seqs = []
        for utterance in self.utterances:
            in_seqs.append(utterance.input_seq_to_use)
        return in_seqs

    def output_seqs(self):
        out_seqs = []
        for utterance in self.utterances:
            out_seqs.append(utterance.gold_query_to_use)
        return out_seqs

def load_function(parameters,
                  nl_to_sql_dict,
                  anonymizer,
                  database_schema=None):
    def fn(interaction_example):
        keep = False

        raw_utterances = interaction_example["interaction"]

        if "database_id" in interaction_example:
            database_id = interaction_example["database_id"]
            interaction_id = interaction_example["interaction_id"]
            identifier = database_id + '/' + str(interaction_id)
        else:
            identifier = interaction_example["id"]

        schema = None
        if database_schema:
            if 'removefrom' not in parameters.data_directory:
                schema = Schema(database_schema[database_id], simple=True)
            else:
                schema = Schema(database_schema[database_id])

        snippet_bank = []

        utterance_examples = []

        anon_tok_to_ent = {}

        for utterance in raw_utterances:
            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1]

            proc_utterance = Utterance(utterance,
                                       available_snippets,
                                       nl_to_sql_dict,
                                       parameters,
                                       anon_tok_to_ent,
                                       anonymizer)
            keep_utterance = proc_utterance.keep

            if schema:
                assert keep_utterance

            if keep_utterance:
                keep = True
                utterance_examples.append(proc_utterance)

                # Update the snippet bank, and age each snippet in it.
                if parameters.use_snippets:
                    if 'atis' in parameters.data_directory:
                        snippets = sql_util.get_subtrees(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)
                    else:
                        snippets = sql_util.get_subtrees_simple(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)

                    for snippet in snippets:
                        snippet.assign_id(len(snippet_bank))
                        snippet_bank.append(snippet)

                for snippet in snippet_bank:
                    snippet.increase_age()

        interaction = Interaction(utterance_examples,
                                  schema,
                                  snippet_bank,
                                  anon_tok_to_ent,
                                  identifier,
                                  parameters)

        return interaction, keep

    return fn
