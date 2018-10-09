import torch
from typing import List

from allennlp.common.checks import ConfigurationError


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    ``combination`` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like ``"1,2,1+2,3-1"``.

    We allow the following kinds of combinations: ``x``, ``x*y``, ``x+y``, ``x-y``, and ``x/y``,
    where ``x`` and ``y`` are positive integers less than or equal to ``len(tensors)``.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the ``combination`` string.  For example, for the input string ``"1,2,1*2"``, the result
    would be ``[1;2;1*2]``, as you would expect, where ``[;]`` is concatenation along the last
    dimension.

    We also allow using a function (avg, abs, sqr, sqrt, halve) on top of the element-wise interaction for examples:
    ``abs(x-y)``, ``halve(x+y)``.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like ``torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])``.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts ``x`` and ``y`` in place of ``1`` and ``2`` in the combination
    string.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(',')]

    return torch.cat(to_concatenate, dim=-1)


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        func = None
        if "(" in combination:
            if ")" not in combination:
                raise ConfigurationError("Closing bracket was not found in {0}".format(combination))

            func_str, combination = tuple(combination.replace(")", "").split("("))

            if func_str == "abs":
                func = torch.abs
            elif func_str == "sqrt":
                func = torch.sqrt
            elif func_str == "sqr":
                func = lambda x: x*x
            elif func_str == "halve":
                func = lambda x: x/2
            else:
                raise ConfigurationError("Invalid function `{0}`! Allowed functions are [abs, sqrt, sqr, halve]!".format(func_str))

        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]

        result = None
        if operation == '*':
            result = first_tensor * second_tensor
        elif operation == '/':
            result = first_tensor / second_tensor
        elif operation == '+':
            result = first_tensor + second_tensor
        elif operation == '-':
            result = first_tensor - second_tensor
        else:
            raise ConfigurationError("Invalid operation: " + operation)

        if func is not None:
            result = func(result)

        return result

def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with :func:`combine_tensors`.  This function computes the resultant dimension when
    calling ``combine_tensors(combination, tensors)``, when the tensor dimension is known.  This is
    necessary for knowing the sizes of weight matrices when building models that use
    ``combine_tensors``.

    Parameters
    ----------
    combination : ``str``
        A comma-separated list of combination pieces, like ``"1,2,1*2"``, specified identically to
        ``combination`` in :func:`combine_tensors`.
    tensor_dims : ``List[int]``
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to :func:`combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')

    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if "(" in combination:
            # handles cases like combination="abs(x-y)"
            if ")" not in combination:
                raise ConfigurationError("Closing bracket was not found in {0}".format(combination))

            combination = combination.replace(")", "").split("(")[-1]

        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)

        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ConfigurationError("Tensor dims must match for operation \"{}\"".format(operation))
        return first_tensor_dim