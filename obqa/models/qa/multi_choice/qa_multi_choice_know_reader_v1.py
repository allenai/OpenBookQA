import numpy
from allennlp.data.dataset import Batch

from allennlp.modules.matrix_attention import LegacyMatrixAttention
from typing import Dict, Optional, List, Any

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, SimilarityFunction
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from obqa.allennlp_custom.utils.common_utils import update_params
from obqa.common.evaluate_predictions_qa_mc_utils import export_output_data_arc_multi_choice_json
from obqa.nn.knowledge import attention_interaction_combinations, embedd_encode_and_aggregate_text_field, \
    embedd_encode_and_aggregate_list_text_field



@Model.register("qa_multi_choice_know_reader_v1")
class QAMultiChoiceKnowReader_v1(Model):
    """
    The ``QAMultiChoiceKnowReader_v1`` models the interaction between question and choice
     and set of knowledge facts. The model is inspired by [1] and is described in details in [2].

    The basic outline of this model is to get an embedded representation for the
    question and choice, and their knowledge-enhanced versions. Then it models an interaction between them and use a linear projection to the class dimension + softmax to get a final predictions:

    question_encoded = context_enc(question_words)  # context encoder can be any AllenNLP supported or None
    choice_encoded = context_enc(choice_words)
    facts_list_encoded = context_enc(facts_list_tokens)

    question_aggregate = aggregate_method(question_encoded)  # bs x hs
    choice_aggregate = aggregate_method(choice_encoded)  # bs x choices_cnt x hs
    facts_list_aggregate = aggregate_method(facts_list_encoded) # bs x facts_cnt x hs

    # get knowledge
    question_kn = att_weighted_sum(question_aggregate, facts_list_aggregate)  # bs x hs
    choices_kn = att_weighted_sum(choice_aggregate, facts_list_aggregate)  # bs x choices_cnt x hs

    # combine context and knowledge
    question_ctx_plus_kn = combine_func(question_kn, question_aggregate)
    choice_ctx_plus_kn = combine_func(choice_kn, choice_aggregate)

    inter = lambda r1, r2: W.concat([r1, r2, abs(r1 - r2), r1 * r2)  # bs x 1,  W is a weight matrix with hs x 1

    @ combination of interactions q_ctx, q_kn, q_ctx_plus_kn and ch_ctx, ch_kn, ch_ctx_plus_kn
    inter_ctx_ctx = inter(question_aggregate, choice_aggregate)
    #...
    inter_kn_ctx = inter(question_kn, choice_aggregate)
    inter_kn_kn = inter(question_kn, choice_kn)
    # ...
    inter_ctx_ctx = inter(question_aggregate, choice_aggregate)

    choice_to_question_att = linear_layer([inter_ctx_ctx,..,inter_kn_ctx,inter_kn_kn,..,inter_ctx_ctx])

    # The choice_to_question_att of the four choices are normalized using ``softmax``
    # and the choice with the highest attention is selected as the answer.

    [1] Mihaylov, T., & Frank, A. (2018). Knowledgeable Reader: Enhancing Cloze-Style Reading Comprehension
    with External Commonsense Knowledge. ACL 2018, http://arxiv.org/abs/1805.07858

    [2] Mihaylov, T., & Frank, A. (2018). Knowledgeable Reader: Enhancing Cloze-Style Reading Comprehension
    with External Commonsense Knowledge. ACL 2018, http://arxiv.org/abs/1805.07858
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
                 use_knowledge: Optional[bool] = True,
                 facts_encoder: Optional[Seq2SeqEncoder] = None,
                 know_aggregate_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 aggregate_question: Optional[str] = "max",
                 aggregate_choice: Optional[str] = "max",
                 aggregate_facts: Optional[str] = "max",
                 embeddings_dropout_value:Optional[float] = 0.0,
                 share_encoders: Optional[bool] = False,
                 choices_init_from_question_states: Optional[bool] = False,
                 facts_init_from_question_states: Optional[bool] = False,
                 use_ctx2facts_retrieval_map_as_mask: Optional[bool] = False,
                 params=Params) -> None:
        super(QAMultiChoiceKnowReader_v1, self).__init__(vocab)

        # TO DO: AllenNLP does not support statefull RNNS yet..
        init_is_supported = False
        if not init_is_supported and (choices_init_from_question_states or facts_init_from_question_states):
            raise ValueError("choices_init_from_question_states=True or facts_init_from_question_states=True are not supported yet!")
        else:
            self._choices_init_from_question_states = choices_init_from_question_states
            self._facts_init_from_question_states = facts_init_from_question_states

        self._return_question_to_choices_att = False
        self._return_question_to_facts_att = False
        self._return_choice_to_facts_att = False

        self._params = params
        
        self._text_field_embedder = text_field_embedder
        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        self._question_encoder = question_encoder

        # choices encoding
        self._choice_encoder = choice_encoder

        # facts encoding
        self._use_knowledge = use_knowledge
        self._facts_encoder = facts_encoder
        self._use_ctx2facts_retrieval_map_as_mask = use_ctx2facts_retrieval_map_as_mask

        self._know_aggregate_feedforward = know_aggregate_feedforward

        self._question_aggregate = aggregate_question
        self._choice_aggregate = aggregate_choice
        self._facts_aggregate = aggregate_facts


        self._num_labels = vocab.get_vocab_size(namespace="labels")

        question_output_dim = self._text_field_embedder.get_output_dim()
        if self._question_encoder is not None:
            question_output_dim = self._question_encoder.get_output_dim()

        choice_output_dim = self._text_field_embedder.get_output_dim()
        if self._choice_encoder is not None:
            choice_output_dim = self._choice_encoder.get_output_dim()

        facts_output_dim = self._text_field_embedder.get_output_dim()
        if self._facts_encoder is not None:
            facts_output_dim = self._facts_encoder.get_output_dim()

        if question_output_dim != choice_output_dim:
            raise ConfigurationError("Output dimension of the question_encoder (dim: {}), "
                                     "plus choice_encoder (dim: {})"
                                     "must match! "
                                     .format(question_output_dim,
                                             choice_output_dim))


        # question to choice attention
        att_question_to_choice_params = params.get("att_question_to_choice")
        if "tensor_1_dim" in att_question_to_choice_params:
            att_question_to_choice_params = update_params(att_question_to_choice_params, {"tensor_1_dim": question_output_dim,
                                                                                                 "tensor_2_dim":choice_output_dim})
        self._matrix_attention_question_to_choice = LegacyMatrixAttention(SimilarityFunction.from_params(att_question_to_choice_params))

        # text to knowlegde attention
        share_att_question_to_choice_and_att_text_to_facts = params.get("share_att_question_to_choice_and_att_text_to_facts", False)
        if share_att_question_to_choice_and_att_text_to_facts:
            self._matrix_attention_text_to_facts = self._matrix_attention_question_to_choice
        else:
            att_text_to_facts_params = params.get("att_text_to_facts")
            if "tensor_1_dim" in att_text_to_facts_params:
                att_text_to_facts_params = update_params(att_text_to_facts_params, {"tensor_1_dim": question_output_dim,
                                                                                    "tensor_2_dim": facts_output_dim})
            self._matrix_attention_text_to_facts = LegacyMatrixAttention(SimilarityFunction.from_params(att_text_to_facts_params))

        text_plus_knowledge_repr_params = params.get("text_plus_knowledge_repr")
        self._text_plus_knowledge_repr_funciton = SimilarityFunction.from_params(text_plus_knowledge_repr_params)

        self._know_interactions = params.get("know_interactions").get("interactions", [["ctx", "ctx"], ["ctx+kn", "ctx"], ["ctx", "ctx+kn"], ["ctx+kn", "ctx+kn"]])

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)


    # propeties

    @property
    def return_question_to_choices_att(self,):
        return self._return_question_to_choices_att
    
    @return_question_to_choices_att.setter
    def return_question_to_choices_att(self, value: bool):
        """
        This makes the model to return question to choice attentions
        :return: nothing
        """

        self._return_question_to_choices_att = value

    @property
    def return_question_to_facts_att(self, ):
        return self._return_question_to_facts_att

    @return_question_to_facts_att.setter
    def return_question_to_facts_att(self, value: bool):
        """
        This makes the model to return question to facts attentions
        :return: nothing
        """

        self._return_question_to_facts_att = value
        
    @property
    def return_choice_to_facts_att(self, ):
        return self._return_choice_to_facts_att

    @return_choice_to_facts_att.setter
    def return_choice_to_facts_att(self, value: bool):
        """
        This makes the model to return question to choice attentions
        :return: nothing
        """

        self._return_choice_to_facts_att = value

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                choices_list: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                facts_list: Dict[str, torch.LongTensor] = None,
                question2facts_map: Dict[str, torch.LongTensor] = None,
                choice2facts_map: Dict[str, torch.LongTensor] = None,
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

        encoded_question_aggregated, question_last_hidden_states = embedd_encode_and_aggregate_text_field(question, self._text_field_embedder, self._embeddings_dropout, self._question_encoder, self._question_aggregate, get_last_states=(self._choices_init_from_question_states or self._facts_init_from_question_states))  # bs, hs

        encoded_choices_aggregated = embedd_encode_and_aggregate_list_text_field(choices_list, self._text_field_embedder, self._embeddings_dropout, self._choice_encoder, self._choice_aggregate, init_hidden_states=question_last_hidden_states if self._choices_init_from_question_states else None)  # # bs, choices, hs

        bs = encoded_question_aggregated.shape[0]
        choices_cnt = encoded_choices_aggregated.shape[1]

        if self._use_knowledge:
            # encode facts
            encoded_facts_aggregated = embedd_encode_and_aggregate_list_text_field(facts_list, self._text_field_embedder, self._embeddings_dropout, self._facts_encoder, self._facts_aggregate, init_hidden_states=question_last_hidden_states if self._facts_init_from_question_states else None)  # # bs, choices, hs

            facts_aggregated_mask = get_text_field_mask(facts_list, num_wrapping_dims=0).float()

            facts_aggregated_mask_q_to_facts = facts_aggregated_mask
            if self._use_ctx2facts_retrieval_map_as_mask and self.training:
                facts_aggregated_mask_q_to_facts = facts_aggregated_mask_q_to_facts * question2facts_map
                facts_aggregated_mask_q_to_facts = (facts_aggregated_mask_q_to_facts > 0.00).float()

            facts_cnt = encoded_facts_aggregated.shape[1]

            # question to knowledge

            q_to_facts_att = self._matrix_attention_text_to_facts(encoded_question_aggregated.unsqueeze(1),
                                                                  encoded_facts_aggregated).view([bs, facts_cnt])
            q_to_facts_att_softmax = util.last_dim_softmax(q_to_facts_att, facts_aggregated_mask_q_to_facts)
            q_to_facts_weighted_sum = util.weighted_sum(encoded_facts_aggregated, q_to_facts_att_softmax)

            assert encoded_question_aggregated.shape == q_to_facts_weighted_sum.shape

            # choices to knowledge

            choices_to_facts_att = self._matrix_attention_text_to_facts(encoded_choices_aggregated,
                                                                        encoded_facts_aggregated).view([bs, choices_cnt, facts_cnt])  # bs, choices, facts

            facts_aggregated_mask_ch_to_facts = facts_aggregated_mask.unsqueeze(1).expand(choices_to_facts_att.shape)
            if self._use_ctx2facts_retrieval_map_as_mask and self.training:
                facts_aggregated_mask_ch_to_facts = facts_aggregated_mask_ch_to_facts * choice2facts_map
                facts_aggregated_mask_ch_to_facts = (facts_aggregated_mask_ch_to_facts > 0.00).float()

            choices_to_facts_att_softmax = util.last_dim_softmax(choices_to_facts_att, facts_aggregated_mask_ch_to_facts)
            choices_to_facts_weighted_sum = util.weighted_sum(encoded_facts_aggregated, choices_to_facts_att_softmax)

            assert encoded_choices_aggregated.shape == choices_to_facts_weighted_sum.shape

            # combine with knowledge
            question_ctx_plus_know = self._text_plus_knowledge_repr_funciton(q_to_facts_weighted_sum, encoded_question_aggregated)
            choices_ctx_plus_know = self._text_plus_knowledge_repr_funciton(choices_to_facts_weighted_sum, encoded_choices_aggregated)

            # question to choices interactions
            q_to_choices_att_list = []

            q_to_choices_combined_att = attention_interaction_combinations(quest_ctx=encoded_question_aggregated,
                                                                           choices_ctx=encoded_choices_aggregated,
                                                                           quest_ctx_plus_kn=question_ctx_plus_know,
                                                                           choices_ctx_plus_kn=choices_ctx_plus_know,
                                                                           quest_kn=q_to_facts_weighted_sum,
                                                                           choices_kn=choices_to_facts_weighted_sum,
                                                                           inter_to_include=self._know_interactions,
                                                                           att_matrix_mappings=self._matrix_attention_question_to_choice)

            # q_to_choices_att = self._matrix_attention_question_to_choice(encoded_question_aggregated.unsqueeze(1),
            #                                                              encoded_choices_aggregated).squeeze()

            if q_to_choices_combined_att.shape[-1] > 1:
                q_to_choices_att = self._know_aggregate_feedforward(q_to_choices_combined_att).squeeze(-1)
            else:
                q_to_choices_att = q_to_choices_combined_att.squeeze(-1)
        else:
            # dont use knowledge
            q_to_choices_att = self._matrix_attention_question_to_choice(encoded_question_aggregated.unsqueeze(1),
                                                                     encoded_choices_aggregated).squeeze()
            # print("No knowledge is used")
        label_logits = q_to_choices_att
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits.data.tolist(), "label_probs": label_probs.data.tolist()}

        if self._return_question_to_choices_att \
                or self._return_question_to_facts_att \
                or self._return_choice_to_facts_att:

            attentions_dict = {}
            know_interactions_weights_dict = {}
            if self._return_question_to_choices_att:
                # Keep also the interaction weights used for the final prediction

                # attentions
                att_to_export_q_to_ch = {}
                q_to_ch_raw_type = "__".join(["ctx", "ctx"])
                if self._use_knowledge:

                    try:
                        # Get interaction weights.
                        # These are currently static but can be replaced with dynamic gating later.
                        know_interactions_weights = self._know_aggregate_feedforward._linear_layers[0].weight.data.tolist()[0]
                    except:
                        know_interactions_weights = [0.0] * len(self._know_interactions)

                    q_to_choices_combined_att_transposed = torch.nn.functional.softmax(q_to_choices_combined_att.permute([2, 0, 1]), dim=-1)

                    # Get the interaction attentions
                    for inter_id, interaction in enumerate(self._know_interactions):
                        interaction_name = "__".join(interaction)
                        att_to_export_q_to_ch[interaction_name] = q_to_choices_combined_att_transposed[inter_id].data.tolist()
                        know_interactions_weights_dict[interaction_name] = know_interactions_weights[inter_id]

                    # In this case "ctx__ctx" is not included in the knowledge interactions for the final prediction,
                    # so we set the weight to 0.0
                    if q_to_ch_raw_type not in know_interactions_weights_dict:
                        know_interactions_weights_dict[q_to_ch_raw_type] = 0.0
                else:
                    # In this case we do not use multiple interactions and the only prediction is for ctx__ctx
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

            if self._use_knowledge:
                if self._return_question_to_facts_att:
                    att_to_export_q_to_f = {}

                    # TO DO: Update when more sources are added
                    att_to_export_q_to_f["src1"] = q_to_facts_att_softmax.data.tolist()
                    attentions_dict["att_q_to_f"] = att_to_export_q_to_f

                if self._return_choice_to_facts_att:
                    att_to_export_ch_to_f = {}

                    # TO DO: Update when more sources are added
                    att_to_export_ch_to_f["src1"] = choices_to_facts_att_softmax.data.tolist()
                    attentions_dict["att_ch_to_f"] = att_to_export_ch_to_f

            output_dict["attentions"] = attentions_dict
            output_dict["know_inter_weights"] = know_interactions_weights_dict

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.long().view([bs]))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QAMultiChoiceKnowReader_v1':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab, embedder_params)

        embeddings_dropout_value = params.pop("embeddings_dropout", 0.0)

        # whether we want to use knowledge
        use_knowledge = params.pop("use_knowledge", True)
        use_ctx2facts_retrieval_map_as_mask = params.pop("use_ctx2facts_retrieval_map_as_mask", False)

        # question encoder
        question_encoder_params = params.pop("question_encoder", None)
        question_enc_aggregate = params.pop("question_encoder_aggregate", "max")
        share_encoders = params.pop("share_encoders", False)

        # condition the choices or facts encoding on quesiton output states
        choices_init_from_question_states = params.pop("choices_init_from_question_states", False)
        facts_init_from_question_states = params.pop("facts_init_from_question_states", False)

        if question_encoder_params is not None:
            question_encoder = Seq2SeqEncoder.from_params(question_encoder_params)
        else:
            question_encoder = None

        knowledge_encoder = None
        knowledge_enc_aggregate = "max"

        if share_encoders:
            choice_encoder = question_encoder
            choice_enc_aggregate = question_enc_aggregate

            if use_knowledge:
                knowledge_encoder = question_encoder
                knowledge_enc_aggregate = question_enc_aggregate
        else:
            # choice encoder
            choice_encoder_params = params.pop("choice_encoder", None)
            choice_enc_aggregate = params.pop("choice_encoder_aggregate", "max")

            if choice_encoder_params is not None:
                choice_encoder = Seq2SeqEncoder.from_params(choice_encoder_params)
            else:
                choice_encoder = None

            if use_knowledge:
                knowledge_encoder_params = params.pop("knowledge_encoder", None)
                knowledge_enc_aggregate = params.pop("knowledge_encoder_aggregate", "max")

                if knowledge_encoder_params is not None:
                    knowledge_encoder = Seq2SeqEncoder.from_params(knowledge_encoder_params)
                else:
                    knowledge_encoder = None

        know_interactions_params = params.get("know_interactions")
        know_interactions_aggregate_ffw_params = know_interactions_params.get('aggregate_feedforward')

        # aggregate knowledge input state is inferred automatically
        update_params(know_interactions_aggregate_ffw_params, {"input_dim": len(know_interactions_params.get("interactions", []))})
        know_aggregate_feedforward = FeedForward.from_params(params.get("know_interactions").get('aggregate_feedforward'))

        init_params = params.pop('initializer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   question_encoder=question_encoder,
                   choice_encoder=choice_encoder,
                   use_knowledge=use_knowledge,
                   facts_encoder=knowledge_encoder,
                   know_aggregate_feedforward=know_aggregate_feedforward,
                   initializer=initializer,
                   aggregate_choice=choice_enc_aggregate,
                   aggregate_question=question_enc_aggregate,
                   aggregate_facts=knowledge_enc_aggregate,
                   embeddings_dropout_value=embeddings_dropout_value,
                   share_encoders=share_encoders,
                   choices_init_from_question_states=choices_init_from_question_states,
                   facts_init_from_question_states=facts_init_from_question_states,
                   use_ctx2facts_retrieval_map_as_mask=use_ctx2facts_retrieval_map_as_mask,
                   params=params)



    def forward_on_instances(self,
                             instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.

        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.

        Returns
        -------
        A list of the models output for each instance.
        """
        with torch.no_grad():
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = dataset.as_tensor_dict()
            outputs = self.decode(self(**model_input))

            instance_separated_output = []

            metadata = [x.fields["metadata"].metadata for x in dataset.instances]
            for res in export_output_data_arc_multi_choice_json(metadata, outputs):
                instance_separated_output.append(res)

            return instance_separated_output
