from allennlp.modules import FeedForward
from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from obqa.allennlp_custom.nn.util import get_combined_dim, combine_tensors

from obqa.allennlp_custom.utils.common_utils import update_params


@SimilarityFunction.register("linear_extended_ffw_repr_comb")
class LinearExtendedFeedForwardReprCombination(SimilarityFunction):
    """
    This similarity function applies a feed-forward layer over
    a combination of the two input vectors, followed by an (optional) activation function.  The
    combination used and the feed-forward layer are configurable.
    The output of the function is the size of the last layer of the FFW. Thus, it can be used also as a configurable
    transformation layer!

    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.

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
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 feedforward_params=None
                 ) -> None:
        super(LinearExtendedFeedForwardReprCombination, self).__init__()
        self._combination = combination

        # aggregate knowledge input state is inferred automatically
        combined_dim = get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])

        update_params(feedforward_params,
                      {"input_dim": combined_dim})
        self._feedforward_layer = FeedForward.from_params(feedforward_params)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = combine_tensors(self._combination, [tensor_1, tensor_2])
        return self._feedforward_layer(combined_tensors)

    @classmethod
    def from_params(cls, params: Params) -> 'LinearExtendedFeedForwardReprCombination':
        tensor_1_dim = params.pop_int("tensor_1_dim")
        tensor_2_dim = params.pop_int("tensor_2_dim")
        combination = params.pop("combination", "x,y")

        feedforward_params = params.pop("feedforward")

        params.assert_empty(cls.__name__)
        return cls(tensor_1_dim=tensor_1_dim,
                   tensor_2_dim=tensor_2_dim,
                   combination=combination,
                   feedforward_params=feedforward_params)
