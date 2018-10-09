import torch
from allennlp.common import Registrable, Params
from allennlp.modules import SimilarityFunction
from allennlp.nn import Activation
from typing import List

from torch.nn import Parameter


@SimilarityFunction.register("linear_transform_sum")
class LinearTransformSumReprCombination(SimilarityFunction):
    """
    This function applies linear transformation for each of the input tensors and takes the sum.
    If output_dim is 0, the dimensions of tensor_1_dim and tensor_2_dim of the two input tensors are expected
    to be equal and the output_dim is set to be their size. This is used since we might want to automatically infer
    the size of the output layer from automatically set values for tensor1 without explicitly knowing the semantic of
    the similarity function.

    Then the output is `W1x + W2y`` where W1 and W2 are linear transformation matrices.

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
        The dimension of the output tensor.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor1_dim: int,
                 tensor2_dim: int,
                 output_dim: int,
                 activation: Activation = None
                 ):

        super().__init__()

        self._tensor1_dim = tensor1_dim
        self._tensor2_dim = tensor2_dim

        self.weight_tensor1 = Parameter(torch.Tensor(tensor1_dim, output_dim))
        self.weight_tensor2 = Parameter(torch.Tensor(tensor2_dim, output_dim))

        if output_dim == 0:
            if tensor1_dim != tensor2_dim:
                raise ValueError("tensor1_dim and tensor2_dim should be equal for this function")

            self._output_dim = tensor1_dim

        self._activation = activation


    def forward(self, tensor1:torch.LongTensor, tensor2:torch.LongTensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Transforms both tensor to a target output dimensions and returns a sum tensor with same
        number of dimensions, such as ``(batch_size, length, out_dim)``.
        """

        res_repr = torch.matmul(tensor1, self.weight_tensor1) \
                   + torch.matmul(tensor2, self.weight_tensor2)

        if self._activation is not None:
            res_repr = self._activation(res_repr)

        return res_repr


    @classmethod
    def from_params(cls, params: Params) -> 'LinearTransformSumReprCombination':
        tensor1_dim = params.get("tensor_1_dim", 0)
        tensor2_dim = params.get("tensor_2_dim", 0)
        output_dim = params.get("output_dim", 0)

        activation = Activation.by_name(params.get("activation", "linear"))()
        return cls(tensor1_dim, tensor2_dim, output_dim, activation)



