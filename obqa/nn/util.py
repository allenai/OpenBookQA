import torch
from allennlp.nn.util import get_final_encoder_states


def seq2vec_seq_aggregate(seq_tensor, mask, aggregate, bidirectional, dim=1):
    """
        Takes the aggregation of sequence tensor

        :param seq_tensor: Batched sequence requires [batch, seq, hs]
        :param mask: binary mask with shape batch, seq_len, 1
        :param aggregate: max, avg, sum
        :param dim: The dimension to take the max. for batch, seq, hs it is 1
        :return:
    """

    seq_tensor_masked = seq_tensor * mask.unsqueeze(-1)
    aggr_func = None
    if aggregate == "last":
        seq = get_final_encoder_states(seq_tensor, mask, bidirectional)
    elif aggregate == "max":
        aggr_func = torch.max
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "min":
        aggr_func = torch.min
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "sum":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "avg":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
        seq_lens = torch.sum(mask, dim=dim)  # this returns batch_size, 1
        seq = seq / seq_lens.view([-1, 1])

    return seq