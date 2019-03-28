from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values

from allennlp.modules.matrix_attention import LegacyMatrixAttention
from typing import Dict, Optional, List, Any

import torch
from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, SimilarityFunction
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from obqa.allennlp_custom.nn.variational_dropout import VariationalDropout


@Model.register("qa_multi_choice_esim")
class QAMultiChoiceESIM(Model):
    """
    This ``Model`` implements a version of ESIM [1], applied to multi-choice question answering setting.
    This is done by modeling the similarity between the question and each of the choices following ESIM.
    As an answer is selected the choice with the highest similarity with the question.

    [1] Enhanced LSTM for Natural Language Inference Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``choice`` ``TextFields`` we get as input to the
        model.
    question_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the question, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    choice_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the choice, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``question_encoder`` for the encoding (doing nothing if ``question_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    embeddings_dropout_value : ``float``, optional (default=0.0)
        The dropout rate used after the encodings layer. If set, it is used only during training.
    embeddings_dropout_value : ``float``, optional (default=0.0)
        The dropout rate used after the embeddings layer. If set, it is used only during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Optional[Seq2SeqEncoder],
                 choice_encoder: Optional[Seq2SeqEncoder],
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 embeddings_dropout_value: Optional[float] = 0.0,
                 encoder_dropout_value: Optional[float] = 0.0,
                 ) -> None:
        super(QAMultiChoiceESIM, self).__init__(vocab)

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        check_dimensions_match(choice_encoder.get_output_dim(), question_encoder.get_output_dim(),
                               "choice_encoder output dim", "question_encoder output dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), question_encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(question_encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

        self._use_cuda = (torch.cuda.is_available() and torch.cuda.current_device() >= 0)

        self._text_field_embedder = text_field_embedder
        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        if encoder_dropout_value:
            self.dropout = torch.nn.Dropout(encoder_dropout_value)
            self.rnn_input_dropout = VariationalDropout(encoder_dropout_value)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._question_encoder = question_encoder

        # choices encoding
        self._choice_encoder = choice_encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        question_output_dim = self._text_field_embedder.get_output_dim()
        if self._question_encoder is not None:
            question_output_dim = self._question_encoder.get_output_dim()

        choice_output_dim = self._text_field_embedder.get_output_dim()
        if self._choice_encoder is not None:
            choice_output_dim = self._choice_encoder.get_output_dim()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                choices_list: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``
        choices_list : Dict[str, torch.LongTensor]
            From a ``List[TextField]``
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

        # encoded_choices_aggregated = embedd_encode_and_aggregate_list_text_field(choices_list,
        #                                                                          self._text_field_embedder,
        #                                                                          self._embeddings_dropout,
        #                                                                          self._choice_encoder,
        #                                                                          self._choice_aggregate)  # # bs, choices, hs
        # 
        # encoded_question_aggregated, _ = embedd_encode_and_aggregate_text_field(question, self._text_field_embedder,
        #                                                                         self._embeddings_dropout,
        #                                                                         self._question_encoder,
        #                                                                         self._question_aggregate,
        #                                                                         get_last_states=False)  # bs, hs
        # 
        # q_to_choices_att = self._matrix_attention_question_to_choice(encoded_question_aggregated.unsqueeze(1),
        #                                                              encoded_choices_aggregated).squeeze()
        # 
        # label_logits = q_to_choices_att
        # label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        # 
        # output_dict = {"label_logits": label_logits, "label_probs": label_probs}
        # 
        # if label is not None:
        #     loss = self._loss(label_logits, label.long().view(-1))
        #     self._accuracy(label_logits, label.squeeze(-1))
        #     output_dict["loss"] = loss

        embedded_question = self._text_field_embedder(question)
        embedded_choices = self._text_field_embedder(choices_list)
        question_mask = get_text_field_mask(question).float()
        choices_mask_3d = get_text_field_mask(choices_list, num_wrapping_dims=1).float()

        # apply dropout for LSTM
        if self._embeddings_dropout:
            embedded_question = self._embeddings_dropout(embedded_question)
            embedded_choices = self._embeddings_dropout(embedded_choices)

        batch_size, choices_cnt, choices_tokens_cnt, emb_size = tuple(embedded_choices.shape)
        choices_mask_flattened = choices_mask_3d.view([batch_size * choices_cnt, choices_tokens_cnt])

        # Shape: (batch_size * choices_cnt, choices_tokens_cnt, embedding_size)
        embedded_choices_flattened = embedded_choices.view([batch_size * choices_cnt, choices_tokens_cnt, -1])

        # encode question and choices

        # Shape: (batch_size, question_tokens_cnt, encoder_out_size)
        encoded_question = self._question_encoder(embedded_question, question_mask)
        question_tokens_cnt = encoded_question.shape[1]
        encoder_out_size = encoded_question.shape[2]

        # tile to choices tokens
        # Shape: (batch_size, choices_cnt, question_tokens_cnt, encoder_out_size)
        encoded_question = encoded_question.unsqueeze(1).expand(batch_size,
                                                                choices_cnt,
                                                                question_tokens_cnt,
                                                                encoder_out_size).contiguous()

        # Shape: (batch_size * choices_cnt, question_tokens_cnt, encoder_out_size)
        encoded_question = encoded_question.view([batch_size * choices_cnt,
                                                  question_tokens_cnt,
                                                  encoder_out_size]).contiguous()

        # tile to choices tokens
        # Shape: (batch_size, choices_cnt, question_length)
        question_mask = question_mask.unsqueeze(1).expand(batch_size,
                                                          choices_cnt,
                                                          question_tokens_cnt).contiguous()

        # Shape: (batch_size * choices_cnt, question_length)
        question_mask = question_mask.view([batch_size * choices_cnt,
                                            question_tokens_cnt]).contiguous()

        # encode choices
        # Shape: (batch_size * choices_cnt, choices_tokens_cnt, encoder_out_size)
        encoded_choices = self._choice_encoder(embedded_choices_flattened, choices_mask_flattened)
        choices_mask = choices_mask_flattened

        # Shape: (batch_size * choices_cnt, question_length, choices_length)
        similarity_matrix = self._matrix_attention(encoded_question, encoded_choices)

        # Shape: (batch_size, question_length, choices_length)
        p2h_attention = last_dim_softmax(similarity_matrix, choices_mask)
        # Shape: (batch_size, question_length, embedding_dim)
        attended_choices = weighted_sum(encoded_choices, p2h_attention)

        # Shape: (batch_size, choices_length, question_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), question_mask)
        # Shape: (batch_size, choices_length, embedding_dim)
        attended_question = weighted_sum(encoded_question, h2p_attention)

        # the "enhancement" layer
        question_enhanced = torch.cat(
            [encoded_question, attended_choices,
             encoded_question - attended_choices,
             encoded_question * attended_choices],
            dim=-1
        )
        choices_enhanced = torch.cat(
            [encoded_choices, attended_question,
             encoded_choices - attended_question,
             encoded_choices * attended_question],
            dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_question = self._projection_feedforward(question_enhanced)
        projected_enhanced_choices = self._projection_feedforward(choices_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_question = self.rnn_input_dropout(projected_enhanced_question)
            projected_enhanced_choices = self.rnn_input_dropout(projected_enhanced_choices)
        v_ai = self._inference_encoder(projected_enhanced_question, question_mask)
        v_bi = self._inference_encoder(projected_enhanced_choices, choices_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(
            v_ai, question_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
            v_bi, choices_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * question_mask.unsqueeze(-1), dim=1) / torch.sum(
            question_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * choices_mask.unsqueeze(-1), dim=1) / torch.sum(
            choices_mask, 1, keepdim=True
        )

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        output_hidden = self._output_feedforward(v_all)
        label_logits = self._output_logit(output_hidden)
        label_logits = label_logits.view([batch_size, choices_cnt])
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QAMultiChoiceMaxAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab, embedder_params)

        embeddings_dropout_value = params.pop("embeddings_dropout", 0.0)
        encoder_dropout_value = params.pop("encoder_dropout", 0.0)

        # question encoder
        question_encoder_params = params.pop("question_encoder", None)
        share_encoders = params.pop("share_encoders", False)

        if question_encoder_params is not None:
            question_encoder = Seq2SeqEncoder.from_params(question_encoder_params)
        else:
            question_encoder = None

        if share_encoders:
            choice_encoder = question_encoder
        else:
            # choice encoder
            choice_encoder_params = params.pop("choice_encoder", None)
            if choice_encoder_params is not None:
                choice_encoder = Seq2SeqEncoder.from_params(choice_encoder_params)
            else:
                choice_encoder = None

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        projection_feedforward = FeedForward.from_params(params.pop('projection_feedforward'))
        inference_encoder = Seq2SeqEncoder.from_params(params.pop("inference_encoder"))
        output_feedforward = FeedForward.from_params(params.pop('output_feedforward'))
        output_logit = FeedForward.from_params(params.pop('output_logit'))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   question_encoder=question_encoder,
                   choice_encoder=choice_encoder,
                   embeddings_dropout_value=embeddings_dropout_value,
                   encoder_dropout_value=encoder_dropout_value,
                   similarity_function=similarity_function,
                   projection_feedforward=projection_feedforward,
                   inference_encoder=inference_encoder,
                   output_feedforward=output_feedforward,
                   output_logit=output_logit,
                   initializer=initializer,
                   regularizer=regularizer)
