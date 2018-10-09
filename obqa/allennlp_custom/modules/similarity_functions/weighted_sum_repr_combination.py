import torch
from allennlp.common import Registrable, Params
from allennlp.modules import SimilarityFunction
from allennlp.nn import Activation
from typing import List

@SimilarityFunction.register("weighted_sum")
class WeightedSumReprCombination(SimilarityFunction):
    """
    This function gets a weighted sum of two vectors where the weight is a scalar value.

    The sizes tensor_1_dim and tensor_2_dim of the two input tensors are expected to be equal
    and the output_dim is set to be their size. This is used since we might want to automatically infer
    the size of the output layer from automatically set values for tensor1 without explicitly knowing the semantic of
    the similarity function.

    If the tensors are ``x``(tensor1) and ``y``(tensor1) and the weight is a scalar value ``alpha``,
    the output is ``alpha*x + (1 - alpha)*y`` where ``*`` is elem-wise multiplication.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    output_dim : ``int``
        This is here for compatibility with the signature of other Similarity functions!
        It is set automatically to the size of tensor1.
    keep_context_threshold : ``float``
       The weight (scalar) to multiply the first tensor (tensor1). tensor2 is multiplied by (1 - keep_context_threshold).
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor1_dim: int,
                 tensor2_dim: int,
                 output_dim: int,
                 keep_context_threshold: float,
                 activation: Activation = None
                 ):
        super().__init__()

        self._tensor1_dim = tensor1_dim
        self._tensor2_dim = tensor2_dim

        self._keep_context_threshold = keep_context_threshold

        if tensor1_dim != tensor2_dim:
            raise ValueError("tensor1_dim and tensor2_dim should be equal for this function")

        # the transformation is sum of two tensors so it is equal their size
        self._output_dim = tensor1_dim


        self._activation = activation



    def forward(self, tensor1:torch.LongTensor, tensor2:torch.LongTensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) linear interatction on the final dimension
        and returns a tensor with same number of dimensions, such as ``(batch_size, length, out_dim)``.
        """

        res_repr = tensor1 * self._keep_context_threshold + (1 - self._keep_context_threshold) * tensor2

        if self._activation is not None:
            res_repr = self._activation(res_repr)

        return res_repr

    @classmethod
    def from_params(cls, params: Params) -> 'WeightedSumReprCombination':
        keep_context_threshold = params.get("keep_context_threshold", 0.5)
        tensor1_dim = params.get("tensor1_dim", 0)
        tensor2_dim = params.get("tensor2_dim", 0)
        output_dim = params.get("output_dim", 0)

        activation = Activation.by_name(params.get("activation", "linear"))()
        return cls(tensor1_dim, tensor2_dim, output_dim, keep_context_threshold, activation)


