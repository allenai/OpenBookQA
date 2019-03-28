from typing import Dict, Optional, AnyStr

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from obqa.nn.util import seq2vec_seq_aggregate


@Model.register("stacked_nn_aggregate_custom")
class StackedNNAggregateCustom(Model):
    """
    This ``StackedNNAggregateCustom`` implements an interaction between premise and hypothesis context-encoded representations
    as defined in [1]

    1. Obtain a BiLSTM (configurable) context representation of the token sequences of the
    `premise` and each `hypothesis`.
    2. Get an aggregated (single vector) representations for `premise` and `hypothesis` using element-wise `max` operation.
    3. Predict the output label `linear_layer([u, v, u - v, u * v])`, where `u` and `v` are the representations from Step 2.

    Pseudo-code looks like:

    premise_encoded = context_enc(premise_words)  # context_enc can be any AllenNLP supported or None. Bi-directional LSTM is used
    hypothesis_encoded = context_enc(hypothesis_words)

    premise_aggregate = aggregate_method(premise_encoded) # aggregate_method can be max, min, avg. ``max`` is used.
    hypothesis_aggregate = aggregate_method(hypothesis_encoded)

    inter = concat([premise_aggregate, hypothesis_aggregate, hypothesis_aggregate - premise_aggregate, premise_aggregate
     * hypothesis_aggregate)

    predicted_label = feedforward(inter)

    [1] Conneau, A. et al. (2017) ‘Supervised Learning of
    Universal Sentence Representations from Natural Language Inference Data’.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    aggregate_feedforward : ``FeedForward``
        These feedforward networks are applied to the concatenated result of the
        encoder networks, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    share_encoders : ``bool``, optional (default=``false``)
        Shares the weights of the premise and hypothesis encoders.
    aggregate_premise : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded premise.
    aggregate_hypothesis : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded hypothesis.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 aggregate_feedforward: FeedForward,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 aggregate_premise: Optional[str] = "max",
                 aggregate_hypothesis: Optional[str] = "max",
                 embeddings_dropout_value: Optional[float] = 0.0,
                 share_encoders: Optional[bool] = False) -> None:
        super(StackedNNAggregateCustom, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder

        self._premise_aggregate = aggregate_premise
        self._hypothesis_aggregate = aggregate_hypothesis

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        premise_output_dim = self._text_field_embedder.get_output_dim()
        if self._premise_encoder is not None:
            premise_output_dim = self._premise_encoder.get_output_dim()

        hypothesis_output_dim = self._text_field_embedder.get_output_dim()
        if self._hypothesis_encoder is not None:
            hypothesis_output_dim = self._hypothesis_encoder.get_output_dim()

        if premise_output_dim != hypothesis_output_dim:
            raise ConfigurationError("Output dimension of the premise_encoder (dim: {}), "
                                     "plus hypothesis_encoder (dim: {})"
                                     "must match! "
                                     .format(premise_output_dim,
                                             hypothesis_output_dim))

        if premise_output_dim * 4 != \
                aggregate_feedforward.get_input_dim():
            raise ConfigurationError("The output of aggregate_feedforward input dim ({2})  "
                                     "should be {3} = 4 x {0} ({1} = premise_output_dim == hypothesis_output_dim)!"
                                     .format(premise_output_dim,
                                             hypothesis_output_dim,
                                             aggregate_feedforward.get_input_dim(),
                                             4 * premise_output_dim))

        if aggregate_feedforward.get_output_dim() != self._num_labels:
            raise ConfigurationError("Final output dimension (%d) must equal num labels (%d)" %
                                     (aggregate_feedforward.get_output_dim(), self._num_labels))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_premise = self._embeddings_dropout(embedded_premise)

        embedded_hypothesis = self._text_field_embedder(hypothesis)
        embedded_hypothesis = self._embeddings_dropout(embedded_hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._premise_encoder:
            embedded_premise = self._premise_encoder(embedded_premise, premise_mask)

        embedded_premise = seq2vec_seq_aggregate(embedded_premise, premise_mask, self._premise_aggregate,
                                                 self._premise_encoder.is_bidirectional(), 1)

        if self._hypothesis_encoder:
            embedded_hypothesis = self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask)

        embedded_hypothesis = seq2vec_seq_aggregate(embedded_hypothesis, hypothesis_mask, self._hypothesis_aggregate,
                                                    self._premise_encoder.is_bidirectional(), 1)

        aggregate_input = torch.cat(
            [embedded_premise, embedded_hypothesis, torch.abs(embedded_hypothesis - embedded_premise),
             embedded_hypothesis * embedded_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            labels = label.long().view(-1)
            loss = self._loss(label_logits, labels)
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'StackedNNAggregateCustom':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab, embedder_params)

        embeddings_dropout_value = params.pop("embeddings_dropout", 0.0)

        share_encoders = params.pop("share_encoders", False)

        # premise encoder
        premise_encoder_params = params.pop("premise_encoder", None)
        premise_enc_aggregate = params.pop("premise_encoder_aggregate", "max")
        if premise_encoder_params is not None:
            premise_encoder = Seq2SeqEncoder.from_params(premise_encoder_params)
        else:
            premise_encoder = None

        # hypothesis encoder
        if share_encoders:
            hypothesis_enc_aggregate = premise_enc_aggregate
            hypothesis_encoder = premise_encoder
        else:
            hypothesis_encoder_params = params.pop("hypothesis_encoder", None)
            hypothesis_enc_aggregate = params.pop("hypothesis_encoder_aggregate", "max")

            if hypothesis_encoder_params is not None:
                hypothesis_encoder = Seq2SeqEncoder.from_params(hypothesis_encoder_params)
            else:
                hypothesis_encoder = None

        aggregate_feedforward = FeedForward.from_params(params.pop('aggregate_feedforward'))

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   aggregate_feedforward=aggregate_feedforward,
                   premise_encoder=premise_encoder,
                   hypothesis_encoder=hypothesis_encoder,
                   initializer=initializer,
                   aggregate_hypothesis=hypothesis_enc_aggregate,
                   aggregate_premise=premise_enc_aggregate,
                   embeddings_dropout_value=embeddings_dropout_value,
                   share_encoders=share_encoders)
