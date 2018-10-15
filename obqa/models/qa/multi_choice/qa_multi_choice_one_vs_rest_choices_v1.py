from allennlp.modules.matrix_attention import LegacyMatrixAttention
from typing import Dict, Optional, List, Any

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from obqa.allennlp_custom.utils.common_utils import update_params

from obqa.nn.knowledge import embedd_encode_and_aggregate_text_field, \
    embedd_encode_and_aggregate_list_text_field


@Model.register("qa_multi_choice_one_vs_rest_choices_v1")
class QAMultiChoice_OneVsRest_Choices_v1(Model):
    """
   This ``QAMultiChoice_OneVsRest_Choices_v1`` can have different modes:

   1. Question to Choice:
        If `use_choice_sum_instead_of_question`==False then this is a classifcal BiLSTM maxout model that models the
        interaction between Question and Choice representation.
   2. Choite-To-Choice: If use_choice_sum_instead_of_question==True.
        In this case the `question` representation is replaced with the average of all choices representations.
   3. Choice only: If the function "att_question_to_choice" has setting
      ```
      "att_question_to_choice":{
      "type": "linear_extended",
      "combination": "y",
      }
      ```
      The `combination` for the interaction is set to `y` which means that only the choice is used to predict the answer.


   In more details teh models work in the following way:
    1. Obtain a BiLSTM context representation of the token sequences of the
    `question` and each `choice`.
    2. Get an aggregated (single vector) representations for `question` and `choice` using element-wise `max` operation.
        If use_choice_sum_instead_of_question== True, then `question` == avg(`choice1`, `choice2`.. `choiceN`)
    3. Compute the attention score between `question` and `choice` as  `linear_layer([u, v, u - v, u * v])`,
    where `u` and `v` are the representations from Step 2.
        Here, we can change the attention function from `linear_layer([u, v, u - v, u * v])` to simply `linear_layer(y)`,
        which means that we will use only the choice for the final prediction!
    4. Select as answer the `choice` with the highest attention with the `question`.

    Pseudo-code looks like:

    question_encoded = context_enc(question_words)  # context_enc can be any AllenNLP supported or None. Bi-directional LSTM is used
    choice_encoded = context_enc(choice_words)

    question_aggregate = aggregate_method(question_encoded) # aggregate_method can be max, min, avg. ``max`` is used.
    choice_aggregate = aggregate_method(choice_encoded)

    If use_choice_sum_instead_of_question==True:
        # In this case we have choice-to-chocies interaction
        question_aggregate = (choice1_aggregate + choice2_aggregate + choice3_aggregate + choice4_aggregate) / 4

    If att_question_to_choice.combination=="x,y,x-y,x*y":
        inter = concat([question_aggregate, choice_aggregate, choice_aggregate - question_aggregate, question_aggregate
         * choice_aggregate)
    elif att_question_to_choice.combination=="y":
        # In this case we have choice-only interaction
        inter = choice_aggregate

    choice_to_question_att = linear_layer(inter) # the output is a scalar value for each question-to-choice interaction

    # The choice_to_question_att of the four choices are normalized using ``softmax``
    # and the choice with the highest attention is selected as the answer.

    The model is inspired by the BiLSTM Max-Out model from Conneau, A. et al. (2017) ‘Supervised Learning of
    Universal Sentence Representations from Natural Language Inference Data’.


    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``choice`` ``TextFields`` we get as input to the
        model.
    aggregate_feedforward : ``FeedForward``
        These feedforward networks are applied to the concatenated result of the
        encoder networks, and its output is used as the entailment class logits.
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
    share_encoders : ``bool``, optional (default=``false``)
        Shares the weights of the question and choice encoders.
    aggregate_question : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded question.
    aggregate_choice : ``str``, optional (default=``max``, allowed options [max, avg, sum, last])
        The aggregation method for the encoded choice.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Optional[Seq2SeqEncoder] = None,
                 choice_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 aggregate_question: Optional[str] = "max",
                 aggregate_choice: Optional[str] = "max",
                 embeddings_dropout_value: Optional[float] = 0.0,
                 share_encoders: Optional[bool] = False,
                 choices_init_from_question_states: Optional[bool] = False,
                 use_choice_sum_instead_of_question: Optional[bool] = False,
                 params=Params) -> None:
        super(QAMultiChoice_OneVsRest_Choices_v1, self).__init__(vocab)

        # TO DO: AllenNLP does not support statefull RNNS yet..
        init_is_supported = False
        if not init_is_supported and (choices_init_from_question_states):
            raise ValueError(
                "choices_init_from_question_states=True or facts_init_from_question_states=True are not supported yet!")
        else:
            self._choices_init_from_question_states = choices_init_from_question_states

        self._use_cuda = (torch.cuda.is_available() and torch.cuda.current_device() >= 0)

        self._return_question_to_choices_att = False
        self._use_choice_sum_instead_of_question = use_choice_sum_instead_of_question

        self._params = params

        self._text_field_embedder = text_field_embedder
        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        self._question_encoder = question_encoder

        # choices encoding
        self._choice_encoder = choice_encoder

        self._question_aggregate = aggregate_question
        self._choice_aggregate = aggregate_choice

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        question_output_dim = self._text_field_embedder.get_output_dim()
        if self._question_encoder is not None:
            question_output_dim = self._question_encoder.get_output_dim()

        choice_output_dim = self._text_field_embedder.get_output_dim()
        if self._choice_encoder is not None:
            choice_output_dim = self._choice_encoder.get_output_dim()

        if question_output_dim != choice_output_dim:
            raise ConfigurationError("Output dimension of the question_encoder (dim: {}), "
                                     "plus choice_encoder (dim: {})"
                                     "must match! "
                                     .format(question_output_dim,
                                             choice_output_dim))

        # question to choice attention
        att_question_to_choice_params = params.get("att_question_to_choice")
        if "tensor_1_dim" in att_question_to_choice_params:
            att_question_to_choice_params = update_params(att_question_to_choice_params,
                                                          {"tensor_1_dim": question_output_dim,
                                                           "tensor_2_dim": choice_output_dim})
        self._matrix_attention_question_to_choice = LegacyMatrixAttention(
            SimilarityFunction.from_params(att_question_to_choice_params))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    # propeties

    @property
    def return_question_to_choices_att(self, ):
        return self._return_question_to_choices_att

    @return_question_to_choices_att.setter
    def return_question_to_choices_att(self, value: bool):
        """
        This makes the model to return question to choice attentions
        :return: nothing
        """

        self._return_question_to_choices_att = value

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

        encoded_choices_aggregated = embedd_encode_and_aggregate_list_text_field(choices_list,
                                                                                 self._text_field_embedder,
                                                                                 self._embeddings_dropout,
                                                                                 self._choice_encoder,
                                                                                 self._choice_aggregate)  # # bs, choices, hs

        if not self._use_choice_sum_instead_of_question:
            encoded_question_aggregated, _ = embedd_encode_and_aggregate_text_field(question, self._text_field_embedder,
                                                                                    self._embeddings_dropout,
                                                                                    self._question_encoder,
                                                                                    self._question_aggregate,
                                                                                    get_last_states=False)  # bs, hs

            q_to_choices_att = self._matrix_attention_question_to_choice(encoded_question_aggregated.unsqueeze(1),
                                                                         encoded_choices_aggregated).squeeze()

            label_logits = q_to_choices_att
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        else:
            bs = encoded_choices_aggregated.shape[0]
            choices_cnt = encoded_choices_aggregated.shape[1]

            ch_to_choices_att = self._matrix_attention_question_to_choice(encoded_choices_aggregated,
                                                                          encoded_choices_aggregated)  # bs, ch, ch

            idx = torch.arange(0, choices_cnt, out=torch.cuda.LongTensor() if self._use_cuda else torch.LongTensor())
            ch_to_choices_att[:, idx, idx] = 0.0

            q_to_choices_att = torch.sum(ch_to_choices_att, dim=1)

            label_logits = q_to_choices_att
            label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if self._return_question_to_choices_att:
            attentions_dict = {}
            know_interactions_weights_dict = {}
            if self._return_question_to_choices_att:
                # Keep also the interaction weights used for the final prediction

                # attentions
                att_to_export_q_to_ch = {}
                q_to_ch_raw_type = "__".join(["ctx", "ctx"])

                if q_to_ch_raw_type not in know_interactions_weights_dict:
                    know_interactions_weights_dict[q_to_ch_raw_type] = 1.0

                if not q_to_ch_raw_type in att_to_export_q_to_ch:
                    q_to_ch_att_ctx_ctx = self._matrix_attention_question_to_choice(
                        encoded_question_aggregated.unsqueeze(1),
                        encoded_choices_aggregated).squeeze()
                    q_to_ch_att_ctx_ctx = torch.nn.functional.softmax(q_to_ch_att_ctx_ctx, dim=-1)
                    att_to_export_q_to_ch[q_to_ch_raw_type] = q_to_ch_att_ctx_ctx.data.tolist()

                att_to_export_q_to_ch["final"] = label_probs.data.tolist()
                attentions_dict["att_q_to_ch"] = att_to_export_q_to_ch

            output_dict["attentions"] = attentions_dict

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QAMultiChoice_OneVsRest_Choices_v1':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab, embedder_params)

        embeddings_dropout_value = params.pop("embeddings_dropout", 0.0)

        # question encoder
        question_encoder_params = params.pop("question_encoder", None)
        question_enc_aggregate = params.pop("question_encoder_aggregate", "max")
        share_encoders = params.pop("share_encoders", False)

        # condition the choices or facts encoding on quesiton output states
        choices_init_from_question_states = params.pop("choices_init_from_question_states", False)

        if question_encoder_params is not None:
            question_encoder = Seq2SeqEncoder.from_params(question_encoder_params)
        else:
            question_encoder = None

        if share_encoders:
            choice_encoder = question_encoder
            choice_enc_aggregate = question_enc_aggregate
        else:
            # choice encoder
            choice_encoder_params = params.pop("choice_encoder", None)
            choice_enc_aggregate = params.pop("choice_encoder_aggregate", "max")

            if choice_encoder_params is not None:
                choice_encoder = Seq2SeqEncoder.from_params(choice_encoder_params)
            else:
                choice_encoder = None

        use_choice_sum_instead_of_question = params.get("use_choice_sum_instead_of_question", False)
        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   question_encoder=question_encoder,
                   choice_encoder=choice_encoder,
                   initializer=initializer,
                   aggregate_choice=choice_enc_aggregate,
                   aggregate_question=question_enc_aggregate,
                   embeddings_dropout_value=embeddings_dropout_value,
                   share_encoders=share_encoders,
                   choices_init_from_question_states=choices_init_from_question_states,
                   use_choice_sum_instead_of_question=use_choice_sum_instead_of_question,
                   params=params)
