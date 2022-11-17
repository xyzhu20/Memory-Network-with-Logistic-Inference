import os
import re
import torch
import json
import string
import collections
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from collections import deque
from .config import *
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize


# from config import *
# emnlp original utils
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qid = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SQUADInputFeature(object):
    def __init__(self, qid, input_ids, token_type_ids, attention_mask, p_mask, offset_mapping, \
                 context, start_pos=None, end_pos=None,
                 is_impossible=None, answer=None
                 ):
        self.qid = qid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.p_mask = p_mask
        self.offset_mapping = offset_mapping
        self.context = context
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.is_impossible = is_impossible
        self.answer = answer


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        data_info = {}
        data_info['qid'] = self.features[index].qid
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(self.features[index].token_type_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[index].p_mask, dtype=torch.long)
        data_info['offset_mapping'] = self.features[index].offset_mapping
        data_info['context'] = self.features[index].context
        data_info['answer_dict'] = self.features[index].answer
        data_info['start_pos'] = torch.tensor(self.features[index].start_pos, dtype=torch.long) if \
            self.features[index].start_pos is not None else None
        data_info['end_pos'] = torch.tensor(self.features[index].end_pos, dtype=torch.long) if \
            self.features[index].end_pos is not None else None
        data_info['is_impossible'] = torch.tensor(self.features[index].is_impossible, dtype=torch.float) if \
            self.features[index].is_impossible is not None else None
        return data_info

    def __len__(self):
        return len(self.features)


def _cuda(x):
    if USE_CUDA:
        return x.cuda(device="cuda:" + str(args.cuda))
    else:
        return x


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def convert_index_to_text(offset_mapping, orig_text, start_index, end_index):
    orig_start_idx = offset_mapping[start_index][0]
    orig_end_idx = offset_mapping[end_index][1]
    return orig_text[orig_start_idx: orig_end_idx]


# in some cases the model will extract long sentence whose first tokens equals to the last tokens
def clean_answer(s):
    def _get_max_matched_str(tlist):
        for length in range(1, len(tlist)):
            if s[:length] == s[-length:]:
                return length
        return -1

    token_list = s.split(' ')
    if len(token_list) > 20:
        max_length = _get_max_matched_str(token_list)
        if max_length == -1:
            rtv = s
        else:
            rtv = " ".join(token_list[:max_length])
        return rtv
    return s


def collate_fn(data):
    data_info = {}
    float_type_keys = ['speaker_target']
    for k in data[0].keys():
        data_info[k] = [d[k] for d in data]
    for k in data_info.keys():
        if isinstance(data_info[k][0], torch.Tensor):
            data_info[k] = _cuda(torch.stack(data_info[k]))
        if isinstance(data_info[k][0], dict):
            new_dict = {}
            for id_key in data_info[k][0].keys():
                if data_info[k][0][id_key] is None:
                    new_dict[id_key] = None
                    continue
                id_key_list = [
                    torch.tensor(sub_dict[id_key], dtype=torch.long if id_key not in float_type_keys else torch.float)
                    for sub_dict in data_info[k]]  # (bsz, seqlen)
                id_key_tensor = torch.stack(id_key_list)
                new_dict[id_key] = _cuda(id_key_tensor)
            data_info[k] = new_dict
    return data_info


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    small = False
    if small:
        examples = examples[:100]
    return examples


def convert_examples_to_features(examples, tokenizer, max_length, training=True,
                                 max_utterance_num=14):
    # tokenizer should be a tokenizer that is inherent from PreTrainedTokenizerFast
    def _get_target_span(target_ids: list, input_ids: list, id_list=None, use_rfind=False):
        if id_list is None:
            id_list = [i for i in range(len(input_ids))]
        span_start_index, span_end_index = args.max_length - 1, args.max_length - 1
        for idx in range(len(id_list)):  # sometimes id_list will exceed the length of input_ids
            id_list[idx] = min(len(input_ids) - 1, id_list[idx])
            id_list[idx] = max(0, id_list[idx])
        id_list = list(set(id_list))  # get rid of redundent ids
        for idx in id_list if not use_rfind else id_list[::-1]:
            is_found = False
            if input_ids[idx] == target_ids[0]:
                is_found = True
                for offset in range(1, len(target_ids)):
                    if idx + offset > len(input_ids) - 1:  # out of range
                        is_found = False
                        break
                    if input_ids[idx + offset] != target_ids[offset]:
                        is_found = False
                        break
                if is_found:
                    span_start_index, span_end_index = idx, idx + len(target_ids) - 1
                    break
        span = (span_start_index, span_end_index)
        return span

    def _get_pos_after_tokenize(pos, offset_mapping, start=True):
        for idx, se in enumerate(offset_mapping):
            if se[0] == se[1] == 0:  # skip pad token
                continue
            if pos == se[0] and start or pos == se[1] and not start:
                return idx
        return max_length - 1

    print("Converting examples to features...")
    max_tokens, max_answer_tokens, max_question_tokens = 0, 0, 0

    p_mask_ids = [tokenizer.sep_token_id, tokenizer.eos_token_id, \
                  tokenizer.bos_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]

    total_num, unptr_num, too_long_num = len(examples), 0, 0
    features = []
    all_num, neg_num = 0, 0
    for exp in tqdm(examples):
        answer_text = exp.orig_answer_text
        context = " ".join(exp.doc_tokens)
        context = tokenizer.pad_token + ' ' + context
        question = exp.question_text

        context_max_length = args.max_length - args.question_max_length
        context_length = len(tokenizer.encode(context))  # including [CLS] and [SEP]
        remain_length = context_max_length - context_length
        context += ' '.join([tokenizer.pad_token] * remain_length)
        assert len(tokenizer.encode(context)) >= context_max_length

        question_length = len(tokenizer.encode(question)) - 1  # except the [CLS] and including the [SEP]
        if question_length > args.question_max_length:
            while len(tokenizer.encode(question)) - 1 > args.question_max_length:
                question = question[:-1]
        remain_length = args.question_max_length - question_length
        question += ' '.join([tokenizer.pad_token] * remain_length)
        question_length = len(tokenizer.encode(question)) - 1
        assert question_length == args.question_max_length, question_length

        ids_dict = tokenizer.encode_plus(context, question, padding='max_length', \
                                         truncation=True, max_length=max_length, return_offsets_mapping=True)
        offset_mapping = ids_dict['offset_mapping']
        input_ids = ids_dict['input_ids']
        token_type_ids = ids_dict['token_type_ids']
        attention_mask = ids_dict['attention_mask']
        for i in range(len(attention_mask)):
            if input_ids[i] == tokenizer.pad_token_id:
                attention_mask[i] = 0
        p_mask = [1] * len(input_ids)
        for i in range(len(input_ids)):
            if input_ids[i] in p_mask_ids or token_type_ids[i] == 1:
                p_mask[i] = 0
        text_len = len(tokenizer.encode(context + ' ' + tokenizer.sep_token + ' ' + question))
        if text_len > max_length: too_long_num += 1

        # inference
        if not training:
            f_tmp = SQUADInputFeature(exp.qid, input_ids, token_type_ids, attention_mask, p_mask, \
                                      offset_mapping, context, answer=None)
            features.append(f_tmp)
            continue
        # training
        is_impossible = 1 if answer_text == '' else 0
        start_pos, end_pos = _get_target_span(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_text)), \
                                              input_ids) if not is_impossible else (
            args.max_length - 1, args.max_length - 1)
        if not is_impossible and (start_pos == max_length - 1 or end_pos == max_length - 1):
            unptr_num += 1
            # print(exp.qid)
            continue

        f_tmp = SQUADInputFeature(exp.qid, input_ids, token_type_ids, attention_mask, p_mask, offset_mapping, \
                                  context, start_pos, end_pos, is_impossible, answer=tokenizer.encode(answer_text),
                                  )
        features.append(f_tmp)
        max_tokens = max(max_tokens, text_len)
        max_answer_tokens = max(max_answer_tokens, len(tokenizer.encode(answer_text)))
        max_question_tokens = max(max_question_tokens, len(tokenizer.encode(question)) - 1)

    if training: print("max token length, max_answer_length, max_question_length: ", max_tokens, max_answer_tokens,
                       max_question_tokens)
    return features, total_num, unptr_num, too_long_num


def get_squad_dataset(input_file, save_path, tokenizer, max_length, training=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    postfix = ""
    for type_ in ["train", "dev", "test"]:
        if type_ in input_file:
            postfix = type_
            break

    example_path = os.path.join(save_path, "example_{}_v2.cache".format(postfix))
    if not os.path.exists(example_path):
        examples = read_squad_examples(input_file, is_training=training,version_2_with_negative=True)
        if not args.colab:
            print("Squad Examples saved to " + example_path)
            torch.save(examples, example_path)
    else:
        print("Read squad_{}_examples from cache...".format(postfix))
        examples = torch.load(example_path)
    feature_path = os.path.join(save_path, "feature_{}_v2.cache".format(postfix))
    if not os.path.exists(feature_path):
        features, _, _, _ = convert_examples_to_features(examples, tokenizer, max_length,
                                                         training=training)
        if not args.colab:
            print("Squad Features saved to " + feature_path)
            torch.save(features, feature_path)
    else:
        print("Read squad_{}_features from cache...".format(postfix))
        features = torch.load(feature_path)
    dataset = Dataset(features)
    return dataset


