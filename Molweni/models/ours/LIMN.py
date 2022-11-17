'''
Mmeory + Coattn + SUP
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizerFast
from transformers import XLNetModel, XLNetConfig, XLNetPreTrainedModel, XLNetTokenizerFast
from transformers import ElectraModel, ElectraConfig, ElectraPreTrainedModel, ElectraTokenizerFast
from transformers import BertLayer
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from utils.config import *
from utils.utils import convert_index_to_text, to_list
from models.component.DUMA import CoAttention, MultiHeadedAttention
from models.component.TransformerBlock import Encoder, EncoderLayer, PositionwiseFeedForward
import copy


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["start_index", "end_index", "start_log_prob", "end_log_prob"])

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetModel, XLNetPreTrainedModel, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast)
}
TRANSFORMER_CLASS = {'bert': 'bert', 'xlnet': 'transformer', 'electra': 'electra'}
CLS_INDEXES = {'bert': 0, 'xlnet': -1, 'electra': 0}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class MRCModel(pretrained_model_class):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.impossible_threshold = 0.7
        self.transformer_name = TRANSFORMER_CLASS[args.model_type]
        self.cls_index = CLS_INDEXES[args.model_type]
        self.question_start = args.max_length - args.question_max_length
        self.speaker_mha_layers = args.mha_layer_num
        if args.model_type == 'xlnet': self.question_start -= 1

        if args.model_type == 'bert':
            self.bert = BertModel(config)
            head_num = 12
        elif args.model_type == 'xlnet':
            self.transformer = XLNetModel(config)
        elif args.model_type == 'electra':
            self.electra = ElectraModel(config)
            head_num = 16

        self.sigmoid = nn.Sigmoid()
        self.start_predictor = PoolerStartLogits(config)
        self.end_predictor = PoolerEndLogits(config)
        self.verifier = nn.Linear(config.hidden_size, 1)  # gate verifier
        self.attn_fct = nn.Linear(config.hidden_size * 3, 1)
        self.utter_filter = nn.Linear(config.hidden_size * 4, 1)
        self.speaker_detector = nn.Linear(config.hidden_size * 4, 1)

        for i in range(self.speaker_mha_layers):
            mha = BertLayer(config)
            self.add_module("MHA_{}".format(str(i)), mha)

        self.QA_coattn = CoAttention(head=head_num, input_dim=config.hidden_size, dropout=0.4)  # QA coattn
        self.gate_coattn = CoAttention(head=head_num, input_dim=config.hidden_size, dropout=0.4)  # gate coattn

        self.gate_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Tanh()
        )

        self.QA_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Tanh()
        )

        self.internal_fusion = nn.Sequential(  # internal_fusion
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Tanh()
        )

        self.memory_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Tanh()
        )

        self.memory_encoder = MemoryEncoder(config, dropout=0.3, head=head_num, N=args.ts_num)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            utterance_ids_dict=None,
            speaker_ids_dict=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            is_impossible=None,
            output_attentions=False,
    ):
        # utterance level
        utterance_gather_ids = utterance_ids_dict['utterance_gather_ids']
        utterance_p_mask = utterance_ids_dict['utterance_p_mask']
        utterance_repeat_num = utterance_ids_dict['utterance_repeat_num']
        key_utterance_target = utterance_ids_dict['key_utterance_target']
        # speaker level
        speaker_gather_ids = utterance_ids_dict['utterance_gather_ids']
        speaker_attn_mask = speaker_ids_dict['speaker_attn_mask']  # (batch,max_utter,dim)
        target_speaker_gather_id = speaker_ids_dict['target_speaker_gather_id']
        speaker_target = speaker_ids_dict['speaker_target']
        speaker_target_mask = speaker_ids_dict['speaker_target_mask']

        training = start_pos is not None and end_pos is not None and is_impossible is not None
        transformer = getattr(self, self.transformer_name)
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)

        # speaker hiddenstates
        speaker_hidden_states = transformer_outputs[1 if args.model_type == 'electra' else 2][
            -(self.speaker_mha_layers + 1)]  # (bsz, seqlen, hsz)
        bsz, slen, hsz = hidden_states.size()

        gate_loss_fct = nn.BCEWithLogitsLoss()
        span_loss_fct = CrossEntropyLoss(ignore_index=hidden_states.shape[1] - 1)
        utter_loss_fct = CrossEntropyLoss(ignore_index=14)
        speaker_loss_fct = nn.BCEWithLogitsLoss()

        # coattn dealwith gatelogits
        gate_logits, gate_prob, gate_coattention = self.gate_prediction(hidden_states, attention_mask)

        # deal with utterance prediction
        utter_logits, utter_weights_repeated, utter_weights = self.key_utterance_prediction(hidden_states,
                                                                                            utterance_gather_ids,
                                                                                            utterance_p_mask,
                                                                                            utterance_repeat_num,
                                                                                            gate_coattention)

        speaker_logits, speaker_mha_out = self.speaker_prediction(hidden_states,
                                                                  speaker_hidden_states,
                                                                  speaker_attn_mask,
                                                                  speaker_gather_ids,
                                                                  target_speaker_gather_id,
                                                                  speaker_target_mask)

        # fuse information
        internal_hidden_states = self.internal_fusion(
            torch.cat(
                [hidden_states, speaker_mha_out, hidden_states * speaker_mha_out, hidden_states - speaker_mha_out],
                dim=-1)
        )

        # memory hidden states
        memory_hidden_states = self.memory_encoder.encode(hidden_states, attention_mask)

        fused_hidden_states = self.memory_fusion(
            torch.cat(
                [internal_hidden_states, memory_hidden_states, internal_hidden_states * memory_hidden_states,
                 internal_hidden_states - memory_hidden_states], dim=-1
            )
        )

        # dialogue to question-key_utterance
        coattention_QA = self.utterance_interaction(fused_hidden_states, utterance_gather_ids, attention_mask,
                                                    gate_prob, utter_weights)

        start_logits = self.start_predictor(fused_hidden_states, coattention_QA, p_mask=p_mask)  # (bsz, seqlen, dim)

        if training:
            end_logits = self.end_predictor(fused_hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = span_loss_fct(start_logits, start_pos)
            end_loss = span_loss_fct(end_logits, end_pos)
            gate_loss = gate_loss_fct(gate_logits, is_impossible)
            span_loss = (start_loss + end_loss) / 2
            utter_loss = utter_loss_fct(utter_logits, key_utterance_target)
            speaker_loss = speaker_loss_fct(speaker_logits, speaker_target)
            total_loss = span_loss + gate_loss + utter_loss + speaker_loss

        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            gate_log_probs = self.sigmoid(gate_logits)  # (bsz)
            gate_index = gate_log_probs > self.impossible_threshold  # (bsz)
            gate_log_probs_list = to_list(gate_log_probs)
            gate_index = to_list(gate_index)

            speaker_index = self.sigmoid(speaker_logits) > 0.5  # (bsz)
            correct_num = ((speaker_index == speaker_target.long()) == speaker_target_mask).sum().item()
            all_num = speaker_target_mask.sum().item()

            start_log_probs = F.softmax(start_logits, dim=-1) * utter_weights_repeated  # shape (bsz, slen)
            # start_log_probs = F.softmax(start_logits, dim=-1)

            max_start_index, max_start_value = to_list(torch.max(start_log_probs, dim=-1).indices)[0], \
                                               to_list(torch.max(start_log_probs, dim=-1).values)[0]
            coresponding_utter_value = to_list(utter_weights_repeated)[0][max_start_index]
            # print(max_start_value, coresponding_utter_value)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(fused_hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            fused_hidden_states_expanded = fused_hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(fused_hidden_states_expanded, start_states=start_states, p_mask=p_mask)

            end_log_probs = F.softmax(end_logits, dim=1) * utter_weights_repeated.unsqueeze(-1).expand_as(
                end_logits)  # shape (bsz, slen, start_n_top)
            # end_log_probs = F.softmax(end_logits, dim=1)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            na_list = []
            for bidx in range(bsz):
                na_list.append((qid[bidx], gate_log_probs_list[bidx]))
                if self.impossible_threshold != -1 and gate_index[bidx] == 1:
                    answer_list.append((qid[bidx], ''))
                    continue
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                answer_list.append((qid[bidx], best_text))

        outputs = (total_loss, gate_loss, span_loss, utter_loss, speaker_loss) if training else (
            answer_list, na_list, (correct_num, all_num))
        return outputs

    def _compute_attention(self, sentence, query, p_mask=None):  # (bsz, slen, hsz) and (bsz, hsz)
        query = query.unsqueeze(1).expand_as(sentence)
        scores = self.attn_fct(torch.cat([sentence, query, sentence * query], dim=-1)).squeeze(-1)  # (bsz, slen)
        if p_mask is not None:
            scores = scores * p_mask - 1e30 * (1 - p_mask)
        weights = torch.softmax(scores, dim=-1)
        context_vec = sentence.mul(weights.unsqueeze(-1).expand_as(sentence)).sum(1)  # (bsz, hsz)
        return weights, context_vec

    def load_mha_params(self):
        for i in range(self.speaker_mha_layers):
            mha = getattr(self, "MHA_{}".format(str(i)))
            rtv = mha.load_state_dict(
                getattr(self, self.transformer_name).encoder.layer[i - self.speaker_mha_layers].state_dict().copy())
            print(rtv)

    def gate_prediction(self, hidden_states, attention_mask):
        Passage = hidden_states[:, :self.question_start, :]
        Question = hidden_states[:, self.question_start:, :]

        mask_P = attention_mask[:, :self.question_start]  # (bsz,length)
        mask_Q = attention_mask[:, self.question_start:]

        coattn1, coattn2 = self.gate_coattn(Passage, Question, mask_P, mask_Q)  # coattn Passage and Question

        coattention = self.gate_fusion(
            torch.cat([coattn1, coattn2, coattn1 * coattn2, coattn1 - coattn2], dim=-1)
        )

        gate_logits = self.verifier(coattention).squeeze(-1)  # (bsz)
        gate_prob = self.sigmoid(gate_logits)
        return gate_logits, gate_prob, coattention

    def speaker_prediction(self, hidden_states, speaker_hidden_states, speaker_attn_mask, speaker_gather_ids,
                           target_speaker_gather_id, speaker_target_mask):
        bsz, slen, hsz = hidden_states.size()
        hidden_states_detached = speaker_hidden_states.detach()  # (bsz, slen, hsz)
        # hidden_states_detached = speaker_hidden_states
        speaker_attn_mask[:, self.question_start:] = 0  # (bsz, slen)
        speaker_attn_mask = (1 - speaker_attn_mask) * -1e30
        speaker_attn_mask = speaker_attn_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.config.num_attention_heads,
                                                                               slen,
                                                                               -1)  # (bsz, n_heads, slen, slen)
        speaker_mha_out = getattr(self, "MHA_0")(hidden_states_detached, attention_mask=speaker_attn_mask)[
            0]  # (bsz, slen, hsz)

        for i in range(1, self.speaker_mha_layers):
            speaker_mha_out = getattr(self, "MHA_{}".format(str(i)))(speaker_mha_out, speaker_attn_mask)[0]
        speaker_embs = speaker_mha_out.gather(dim=1, index=speaker_gather_ids.unsqueeze(-1).expand(-1, -1,
                                                                                                   hsz))  # (bsz, max_utterm, hsz)
        masked_speaker_embs = speaker_embs.gather(dim=1, index=target_speaker_gather_id.unsqueeze(-1).expand(-1, -1,
                                                                                                             hsz))  # (bsz, 1, hsz)
        masked_speaker_embs_expand = masked_speaker_embs.expand_as(speaker_embs)  # (bsz, max_utter, hsz)
        speaker_logits = self.speaker_detector(
            torch.cat([speaker_embs, masked_speaker_embs_expand, \
                       speaker_embs * masked_speaker_embs_expand, speaker_embs - masked_speaker_embs_expand],
                      dim=-1)
        ).squeeze(-1)  # (bsz, max_utter)
        speaker_logits = speaker_logits * speaker_target_mask - 1e30 * (1 - speaker_target_mask)
        return speaker_logits, speaker_mha_out

    def key_utterance_prediction(self, hidden_states, utterance_gather_ids, utterance_p_mask,
                                 utterance_repeat_num, gate_coattention):
        bsz, slen, hsz = hidden_states.size()

        utter_embs = hidden_states.gather(dim=1, index=utterance_gather_ids.unsqueeze(-1).expand(-1, -1,
                                                                                                 hsz))  # (bsz, max_utter=14, hsz)

        # question_emb_expand = question_emb.unsqueeze(1).expand_as(utter_embs)
        coattention_emb_expand = gate_coattention.unsqueeze(1).expand_as(utter_embs)
        utter_logits = self.utter_filter(torch.cat(
            [utter_embs, coattention_emb_expand, utter_embs * coattention_emb_expand,
             utter_embs - coattention_emb_expand],
            dim=-1)
        ).squeeze(-1)  # (bsz, max_utter)

        utter_logits = utter_logits * utterance_p_mask - 1e30 * (1 - utterance_p_mask)
        utter_weights = torch.softmax(utter_logits, dim=-1)  # (bsz, max_utter)
        utter_weights_repeated = utter_weights.view(-1).repeat_interleave(utterance_repeat_num.view(-1)).view(bsz,
                                                                                                              -1)  # (bsz, slen)

        return utter_logits, utter_weights_repeated, utter_weights

    def utterance_interaction(self, hidden_states, utterance_gather_ids, attention_mask, gate_prob, utter_weights):

        bsz, slen, hsz = hidden_states.size()
        Passage = hidden_states[:, :self.question_start, :]
        Question = hidden_states[:, self.question_start:, :]

        mask_P = attention_mask[:, :self.question_start]  # (bsz,length)
        mask_Q = attention_mask[:, self.question_start:]
        # 取出答案
        utter_index = torch.argmax(utter_weights, dim=-1).unsqueeze(-1)  # (bsz)
        # key_utterance_embs = utter_embs.gather(dim=1, index=utter_index.unsqueeze(-1).expand(-1, -1, hsz))
        # 取出utter对应的token并进行padding
        Answer = []
        mask_A = []
        utter_max_length = 109
        for i in range(bsz):
            if gate_prob[i] > self.impossible_threshold:  # 预测为没有答案
                start = 389
                end = 390
            else:
                if utter_index[i][0] == 0:
                    start = 2
                else:
                    start = utterance_gather_ids[i][utter_index[i][0] - 1] + 1
                end = utterance_gather_ids[i][utter_index[i][0]] + 1
            ans = hidden_states[i, start:end, :]
            mask_ans = attention_mask[i, start:end]

            if (end - start) < utter_max_length:
                pad_num = utter_max_length - (end - start)
                ans_pad = get_cuda(torch.tensor([0] * pad_num).unsqueeze(-1).expand(-1, hsz))
                mask_pad = get_cuda(torch.tensor([0] * pad_num))
                ans = torch.cat([ans, ans_pad], dim=0)
                mask_ans = torch.cat([mask_ans, mask_pad], dim=0)
            elif (end - start) > utter_max_length:
                ans = ans[:utter_max_length, :]
                mask_ans = mask_ans[:utter_max_length]
            Answer.append(ans)
            mask_A.append(mask_ans)
        Answer = torch.stack(Answer)
        mask_A = torch.stack(mask_A)

        QA = torch.cat([Question, Answer], dim=1)
        mask_QA = torch.cat([mask_Q, mask_A], dim=-1)
        coattn_QA_1, coattn_QA_2 = self.QA_coattn(Passage, QA, mask_P, mask_QA)  # coattn with Passage and QA

        coattention_QA = self.QA_fusion(
            torch.cat([coattn_QA_1, coattn_QA_2, coattn_QA_1 * coattn_QA_2, coattn_QA_1 - coattn_QA_2], dim=-1)
        )

        return coattention_QA

    def train_MemEncoder(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            p_mask=None,
            context=None,
            offset_mapping=None,
            qid=None,
            start_pos=None,
            end_pos=None,
            is_impossible=None,
            output_attentions=False,
    ):
        training = start_pos is not None and end_pos is not None and is_impossible is not None
        transformer = get_cuda(getattr(self, self.transformer_name))
        transformer_outputs = transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        hidden_states = transformer_outputs[0]  # (bsz, seqlen, hsz)

        hidden_states = self.memory_encoder(hidden_states, attention_mask)

        squad_gate_loss_fct = nn.BCEWithLogitsLoss()
        squad_span_loss_fct = CrossEntropyLoss(ignore_index=hidden_states.shape[1] - 1)

        # coattn deal with gate logits
        gate_logits, _, gate_coattention = self.gate_prediction(hidden_states, attention_mask)
        start_logits = self.start_predictor(hidden_states, gate_coattention, p_mask=p_mask)  # (bsz, seqlen)

        if training:
            end_logits = self.end_predictor(hidden_states, start_positions=start_pos, p_mask=p_mask)
            start_loss = squad_span_loss_fct(start_logits, start_pos)
            end_loss = squad_span_loss_fct(end_logits, end_pos)
            gate_loss = squad_gate_loss_fct(gate_logits, is_impossible)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss + gate_loss
        else:
            # during inference, compute the end logits based on beam search
            assert context is not None and offset_mapping is not None
            bsz, slen, hsz = hidden_states.size()
            gate_log_probs = self.sigmoid(gate_logits)  # (bsz)
            gate_index = gate_log_probs > self.impossible_threshold  # (bsz)
            gate_log_probs_list = to_list(gate_log_probs)
            gate_index = to_list(gate_index)

            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_predictor(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_top_index = to_list(start_top_index)
            start_top_log_probs = to_list(start_top_log_probs)
            end_top_index = to_list(end_top_index)
            end_top_log_probs = to_list(end_top_log_probs)

            answer_list = []
            na_list = []
            for bidx in range(bsz):
                na_list.append((qid[bidx], gate_log_probs_list[bidx]))
                if self.impossible_threshold != -1 and gate_index[bidx] == 1:
                    answer_list.append((qid[bidx], ''))
                    continue
                b_offset_mapping = offset_mapping[bidx]
                b_orig_text = context[bidx]
                prelim_predictions = []
                for i in range(self.start_n_top):
                    for j in range(self.end_n_top):
                        start_log_prob = start_top_log_probs[bidx][i]
                        start_index = start_top_index[bidx][i]
                        j_index = i * self.end_n_top + j
                        end_log_prob = end_top_log_probs[bidx][j_index]
                        end_index = end_top_index[bidx][j_index]

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)
                best_text = ''
                if len(prelim_predictions) > 0:
                    best_one = prelim_predictions[0]
                    best_start_index = best_one.start_index
                    best_end_index = best_one.end_index
                    best_text = convert_index_to_text(b_offset_mapping, b_orig_text, best_start_index, best_end_index)
                answer_list.append((qid[bidx], best_text))

        outputs = (total_loss, gate_loss, span_loss,) if training else (answer_list, na_list,)
        return outputs


class PoolerStartLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU())
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
            self, hidden_states: torch.FloatTensor,
            question_emb: torch.FloatTensor,  # (bsz, hsz)
            p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        question_emb = question_emb.unsqueeze(1).expand_as(hidden_states)
        x = self.fusion(
            torch.cat([hidden_states, question_emb, hidden_states * question_emb], dim=-1))
        x = self.dense(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1 - p_mask)
            else:
                x = x * p_mask - 1e30 * (1 - p_mask)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            start_states: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        assert (
                start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * p_mask - 65500 * (1 - p_mask)
            else:
                x = x * p_mask - 1e30 * (1 - p_mask)

        return x


class MemoryEncoder(nn.Module):
    """
    N Transformer Layers.
    """

    def __init__(self, config, dropout, head, N):
        super(MemoryEncoder, self).__init__()
        self.config = config
        c = copy.deepcopy
        attn = MultiHeadedAttention(head, d_model=config.hidden_size)
        ff = PositionwiseFeedForward(d_model=config.hidden_size, d_ff=config.hidden_size * 2, dropout=dropout)
        self.encoder = Encoder(EncoderLayer(config.hidden_size, c(attn), c(ff), dropout), N)

    def forward(self, src, src_mask):
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask.unsqueeze(1))
