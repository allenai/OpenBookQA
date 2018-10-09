from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from typing import Union, Dict

import torch
from allennlp.modules import MatrixAttention, Seq2SeqEncoder

from obqa.nn.util import seq2vec_seq_aggregate


def attention_combinations_dict(quest_repr_list,
                                choices_repr_list,
                                inter_to_include,
                                att_matrix_mappings: Union[MatrixAttention, Dict[str, MatrixAttention]]):
    """
    This method calculates interactions between different representations for question and choices.
    :param quest_repr_list:
    :param choices_repr_list:
    :param inter_to_include:
    :param att_matrix_mappings:
    :return:
    """

    att_matrix_mappings_dict = {}
    if isinstance(att_matrix_mappings, MatrixAttention):
        att_matrix_mappings_dict["default"] = att_matrix_mappings
    elif isinstance(att_matrix_mappings, dict):
        att_matrix_mappings_dict = att_matrix_mappings
    else:
        raise ValueError("att_matrix_mappings should be either MatrixAttention, Dict[str, MatrixAttention]")

    interactions = []
    for inter_setting in inter_to_include:
        inter_setting_key = "_".join(inter_setting)
        att_matrix = att_matrix_mappings_dict.get(inter_setting_key, att_matrix_mappings_dict.get("default"))
        curr_inter = att_matrix(quest_repr_list[inter_setting[0]].unsqueeze(1), choices_repr_list[inter_setting[1]]) \
            .squeeze(1)  # last squeeze is to fix the question unsqueeze.

        interactions.append(curr_inter.unsqueeze(-1))

    concat_inter = torch.cat(interactions, dim=-1)

    return concat_inter


def attention_interaction_combinations(quest_ctx,
                                       choices_ctx,
                                       quest_ctx_plus_kn,
                                       choices_ctx_plus_kn,
                                       quest_kn,
                                       choices_kn,
                                       inter_to_include,
                                       att_matrix_mappings: Union[MatrixAttention, Dict[str, MatrixAttention]]):
    """
    This method calculates interactions between different representations for question and choices.
    Currently it is not optimized! Calculated sequentially for type tuples [("ctx", "ctx+kn"), ("ctx+kn", "ctx"), etc.]
    :param quest_ctx:
    :param choices_ctx:
    :param quest_ctx_plus_kn:
    :param choices_ctx_plus_kn:
    :param quest_kn:
    :param choices_kn:
    :param inter_to_include:
    :return: Interactions combinations
    """
    quest_repr_list = {
        "ctx": quest_ctx,
        "ctx+kn": quest_ctx_plus_kn,
        "kn": quest_kn,
    }

    choices_repr_list = {
        "ctx": choices_ctx,
        "ctx+kn": choices_ctx_plus_kn,
        "kn": choices_kn,
    }


    att_matrix_mappings_dict = {}
    if isinstance(att_matrix_mappings, MatrixAttention):
        att_matrix_mappings_dict["default"] = att_matrix_mappings
    elif isinstance(att_matrix_mappings, dict):
        att_matrix_mappings_dict = att_matrix_mappings
    else:
        raise ValueError("att_matrix_mappings should be either MatrixAttention, Dict[str, MatrixAttention]")

    interactions = []
    for inter_setting in inter_to_include:
        inter_setting_key = "_".join(inter_setting)
        att_matrix = att_matrix_mappings_dict.get(inter_setting_key, att_matrix_mappings_dict.get("default"))
        curr_inter = att_matrix(quest_repr_list[inter_setting[0]].unsqueeze(1), choices_repr_list[inter_setting[1]]) \
            .squeeze(1)  # last squeeze is to fix the question unsqueeze.

        interactions.append(curr_inter.unsqueeze(-1))

    concat_inter = torch.cat(interactions, dim=-1)

    return concat_inter


def embedd_encode_and_aggregate_text_field(question: Dict[str, torch.LongTensor],
                                           text_field_embedder,
                                           embeddings_dropout,
                                           encoder,
                                           aggregation_type,
                                           get_last_states=False):
    embedded_question = text_field_embedder(question)
    question_mask = get_text_field_mask(question).float()
    embedded_question = embeddings_dropout(embedded_question)

    encoded_question = encoder(embedded_question, question_mask)

    # aggregate sequences to a single item
    encoded_question_aggregated = seq2vec_seq_aggregate(encoded_question, question_mask, aggregation_type,
                                                        encoder.is_bidirectional(), 1)  # bs X d

    last_hidden_states = None
    if get_last_states:
        last_hidden_states = get_final_encoder_states(encoded_question, question_mask, encoder.is_bidirectional())

    return encoded_question_aggregated, last_hidden_states

def embedd_encode_and_aggregate_list_text_field(texts_list: Dict[str, torch.LongTensor],
                                                text_field_embedder,
                                                embeddings_dropout,
                                                encoder: Seq2SeqEncoder,
                                                aggregation_type,
                                                init_hidden_states=None):
    embedded_texts = text_field_embedder(texts_list)
    embedded_texts = embeddings_dropout(embedded_texts)

    bs, ch_cnt, ch_tkn_cnt, d = tuple(embedded_texts.shape)

    embedded_texts_flattened = embedded_texts.view([bs * ch_cnt, ch_tkn_cnt, -1])
    # masks

    texts_mask_dim_3 = get_text_field_mask(texts_list, num_wrapping_dims=1).float()
    texts_mask_flatened = texts_mask_dim_3.view([-1, ch_tkn_cnt])

    # context encoding
    multiple_texts_init_states = None
    if init_hidden_states is not None:
        if init_hidden_states.shape[0] == bs and init_hidden_states.shape[1] != ch_cnt:
            if init_hidden_states.shape[1] != encoder.get_output_dim():
                raise ValueError("The shape of init_hidden_states is {0} but is expected to be {1} or {2}".format(str(init_hidden_states.shape),
                                                                            str([bs, encoder.get_output_dim()]),
                                                                            str([bs, ch_cnt, encoder.get_output_dim()])))
            # in this case we passed only 2D tensor which is the default output from question encoder
            multiple_texts_init_states = init_hidden_states.unsqueeze(1).expand([bs, ch_cnt, encoder.get_output_dim()]).contiguous()

            # reshape this to match the flattedned tokens
            multiple_texts_init_states = multiple_texts_init_states.view([bs * ch_cnt, encoder.get_output_dim()])
        else:
            multiple_texts_init_states = init_hidden_states.view([bs * ch_cnt, encoder.get_output_dim()])

    encoded_texts_flattened = encoder(embedded_texts_flattened, texts_mask_flatened, hidden_state=multiple_texts_init_states)

    aggregated_choice_flattened = seq2vec_seq_aggregate(encoded_texts_flattened, texts_mask_flatened,
                                                        aggregation_type,
                                                        encoder.is_bidirectional(),
                                                        1)  # bs*ch X d

    aggregated_choice_flattened_reshaped = aggregated_choice_flattened.view([bs, ch_cnt, -1])
    return aggregated_choice_flattened_reshaped


def embed_encode_and_aggregate_text_field_with_feats(question: Dict[str, torch.LongTensor],
                                           text_field_embedder,
                                           embeddings_dropout,
                                           encoder,
                                           aggregation_type,
                                           token_features=None,
                                           get_last_states=False):
    embedded_question = text_field_embedder(question)
    question_mask = get_text_field_mask(question).float()
    embedded_question = embeddings_dropout(embedded_question)
    if token_features is not None:
        embedded_question = torch.cat([embedded_question, token_features], dim=-1)

    encoded_question = encoder(embedded_question, question_mask)

    # aggregate sequences to a single item
    encoded_question_aggregated = seq2vec_seq_aggregate(encoded_question, question_mask, aggregation_type,
                                                        encoder.is_bidirectional(), 1)  # bs X d

    last_hidden_states = None
    if get_last_states:
        last_hidden_states = get_final_encoder_states(encoded_question, question_mask, encoder.is_bidirectional())

    return encoded_question_aggregated, last_hidden_states

def embed_encode_and_aggregate_text_field_with_feats_only(question: Dict[str, torch.LongTensor],
                                           text_field_embedder,
                                           embeddings_dropout,
                                           encoder,
                                           aggregation_type,
                                           token_features=None,
                                           get_last_states=False):
    embedded_question = text_field_embedder(question)
    question_mask = get_text_field_mask(question).float()
    embedded_question = embeddings_dropout(embedded_question)
    if token_features is not None:
        embedded_question = torch.cat([token_features], dim=-1)

    encoded_question = encoder(embedded_question, question_mask)

    # aggregate sequences to a single item
    encoded_question_aggregated = seq2vec_seq_aggregate(encoded_question, question_mask, aggregation_type,
                                                        encoder.is_bidirectional(), 1)  # bs X d

    last_hidden_states = None
    if get_last_states:
        last_hidden_states = get_final_encoder_states(encoded_question, question_mask, encoder.is_bidirectional())

    return encoded_question_aggregated, last_hidden_states

def embed_encode_and_aggregate_list_text_field_with_feats(texts_list: Dict[str, torch.LongTensor],
                                                text_field_embedder,
                                                embeddings_dropout,
                                                encoder: Seq2SeqEncoder,
                                                aggregation_type,
                                                token_features=None,
                                                init_hidden_states=None):
    embedded_texts = text_field_embedder(texts_list)
    embedded_texts = embeddings_dropout(embedded_texts)

    if token_features is not None:
        embedded_texts = torch.cat([embedded_texts, token_features], dim=-1)

    bs, ch_cnt, ch_tkn_cnt, d = tuple(embedded_texts.shape)

    embedded_texts_flattened = embedded_texts.view([bs * ch_cnt, ch_tkn_cnt, -1])
    # masks

    texts_mask_dim_3 = get_text_field_mask(texts_list, num_wrapping_dims=1).float()
    texts_mask_flatened = texts_mask_dim_3.view([-1, ch_tkn_cnt])

    # context encoding
    multiple_texts_init_states = None
    if init_hidden_states is not None:
        if init_hidden_states.shape[0] == bs and init_hidden_states.shape[1] != ch_cnt:
            if init_hidden_states.shape[1] != encoder.get_output_dim():
                raise ValueError("The shape of init_hidden_states is {0} but is expected to be {1} or {2}".format(str(init_hidden_states.shape),
                                                                            str([bs, encoder.get_output_dim()]),
                                                                            str([bs, ch_cnt, encoder.get_output_dim()])))
            # in this case we passed only 2D tensor which is the default output from question encoder
            multiple_texts_init_states = init_hidden_states.unsqueeze(1).expand([bs, ch_cnt, encoder.get_output_dim()]).contiguous()

            # reshape this to match the flattedned tokens
            multiple_texts_init_states = multiple_texts_init_states.view([bs * ch_cnt, encoder.get_output_dim()])
        else:
            multiple_texts_init_states = init_hidden_states.view([bs * ch_cnt, encoder.get_output_dim()])

    encoded_texts_flattened = encoder(embedded_texts_flattened, texts_mask_flatened, hidden_state=multiple_texts_init_states)

    aggregated_choice_flattened = seq2vec_seq_aggregate(encoded_texts_flattened, texts_mask_flatened,
                                                        aggregation_type,
                                                        encoder.is_bidirectional(),
                                                        1)  # bs*ch X d

    aggregated_choice_flattened_reshaped = aggregated_choice_flattened.view([bs, ch_cnt, -1])
    return aggregated_choice_flattened_reshaped


def embed_encode_and_aggregate_list_text_field_with_feats_only(texts_list: Dict[str, torch.LongTensor],
                                                text_field_embedder,
                                                embeddings_dropout,
                                                encoder: Seq2SeqEncoder,
                                                aggregation_type,
                                                token_features=None,
                                                init_hidden_states=None):
    embedded_texts = text_field_embedder(texts_list)
    embedded_texts = embeddings_dropout(embedded_texts)

    if token_features is not None:
        embedded_texts = torch.cat([token_features], dim=-1)

    bs, ch_cnt, ch_tkn_cnt, d = tuple(embedded_texts.shape)

    embedded_texts_flattened = embedded_texts.view([bs * ch_cnt, ch_tkn_cnt, -1])
    # masks

    texts_mask_dim_3 = get_text_field_mask(texts_list, num_wrapping_dims=1).float()
    texts_mask_flatened = texts_mask_dim_3.view([-1, ch_tkn_cnt])

    # context encoding
    multiple_texts_init_states = None
    if init_hidden_states is not None:
        if init_hidden_states.shape[0] == bs and init_hidden_states.shape[1] != ch_cnt:
            if init_hidden_states.shape[1] != encoder.get_output_dim():
                raise ValueError("The shape of init_hidden_states is {0} but is expected to be {1} or {2}".format(str(init_hidden_states.shape),
                                                                            str([bs, encoder.get_output_dim()]),
                                                                            str([bs, ch_cnt, encoder.get_output_dim()])))
            # in this case we passed only 2D tensor which is the default output from question encoder
            multiple_texts_init_states = init_hidden_states.unsqueeze(1).expand([bs, ch_cnt, encoder.get_output_dim()]).contiguous()

            # reshape this to match the flattedned tokens
            multiple_texts_init_states = multiple_texts_init_states.view([bs * ch_cnt, encoder.get_output_dim()])
        else:
            multiple_texts_init_states = init_hidden_states.view([bs * ch_cnt, encoder.get_output_dim()])

    encoded_texts_flattened = encoder(embedded_texts_flattened, texts_mask_flatened, hidden_state=multiple_texts_init_states)

    aggregated_choice_flattened = seq2vec_seq_aggregate(encoded_texts_flattened, texts_mask_flatened,
                                                        aggregation_type,
                                                        encoder.is_bidirectional(),
                                                        1)  # bs*ch X d

    aggregated_choice_flattened_reshaped = aggregated_choice_flattened.view([bs, ch_cnt, -1])
    return aggregated_choice_flattened_reshaped

