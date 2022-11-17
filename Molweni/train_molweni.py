# Molweni batch size: BERT 24 ELECTRA 8

import json
import torch
import numpy as np
import random
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, ElectraTokenizerFast
from transformers import BertConfig, ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluate_v2 import main as evaluate_on_squad, EVAL_OPTS

from utils.config import *
from models.ours import LIMN
from models.baseline import baseline,SUP

from utils.utils import get_dataset, collate_fn

MRC_MODEL_LIST = [baseline, SUP, LIMN]

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast),
}

warnings.filterwarnings("ignore")
device = torch.device("cuda:" + str(args.cuda)) if USE_CUDA else torch.device("cpu")
train_path = os.path.join(args.data_path, "Molweni/train.json")
eval_path = os.path.join(args.data_path, "Molweni/dev.json")
test_path = os.path.join(args.data_path, "Molweni/test.json")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]

MRC_NAME = ['_baseline_', '_SUP_', '_LIMN_']

model_save_path = "./saves/checkpoint/" + args.model_type + MRC_NAME[args.model_num] + "Molweni_" + str(
    args.learning_rate) + "_T" + str(args.ts_num) + "_M" + str(args.mha_layer_num) + '.pkl'

result_json_save_path = "./saves/result/" + args.model_type + MRC_NAME[args.model_num] + "Molweni_" + str(
    args.learning_rate) + "_T" + str(args.ts_num) + "_M" + str(args.mha_layer_num) + '.json'

pretrained_model_path = "./saves/checkpoint/" + args.model_type + "_SQuAD_"+ str(6e-6) + "_T" + str(args.ts_num) + '.pkl'


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def train(model, train_loader, eval_dataloader, test_dataloader, tokenizer):
    save_dev_f1 = 0
    save_dev_em = 0

    print("Traning arguments:")
    print(args)

    patience_turns = 0
    model.train()
    model.zero_grad()
    
    
    # freeze some params not forward
    # freeze = ['internal_fusion', 'QA_fusion', 'memory_fusion', 'QA_coattn', 'speaker_detector', 'utter_filter',
    #           'attn_fct','MHA_']

    # for name, param in model.named_parameters():
    #     if any(nf in name for nf in freeze):
    #         print(name,param.requires_grad)
            
    # freeze memory encoder
    for name, param in model.memory_encoder.named_parameters():
        param.requires_grad = False
        print(name,param.requires_grad)
    print(pretrained_model_path)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    (not any(nd in n for nd in no_decay)) and p.requires_grad == True],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    (any(nd in n for nd in no_decay)) and p.requires_grad == True], 'weight_decay': 0.0}
    ]

    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    logging_step = t_total // (args.epochs * 5)
    steps = 0

    for epoch in range(args.epochs):
        avg_loss, avg_gate_loss, avg_span_loss, avg_utter_loss, avg_speaker_loss = 0, 0, 0, 0, 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask'],
                      'p_mask': batch['p_mask'],
                      'utterance_ids_dict': batch['utterance_ids_dict'],
                      'start_pos': batch['start_pos'],
                      'end_pos': batch['end_pos'],
                      'is_impossible': batch['is_impossible'],
                      }
            if args.add_speaker_mask:
                inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            if t_total is not None:
                scheduler.step()
            if len(outputs) == 5:  # with SUP
                gate_loss, span_loss, utter_loss, speaker_loss = outputs[1].item(), outputs[2].item(), \
                                                                 outputs[3].item(), outputs[4].item()
                pbar.set_description(
                    "Epoch:%d | Loss:%.3f | GateLoss:%.3f | SpanLoss:%.3f | UtterLoss:%.3f | SpeakerLoss:%.3f" \
                    % (epoch, loss.item(), gate_loss, span_loss, utter_loss, speaker_loss))
                avg_loss += loss
                avg_span_loss += span_loss
                avg_gate_loss += gate_loss
                avg_utter_loss += utter_loss
                avg_speaker_loss += speaker_loss
            elif len(outputs) == 3:  # without SUP
                gate_loss, span_loss = outputs[1].item(), outputs[2].item()
                pbar.set_description(
                    "Epoch:%d | Loss:%.3f | GateLoss:%.3f | SpanLoss:%.3f " % (
                        epoch, loss.item(), gate_loss, span_loss))
                avg_loss += loss
                avg_span_loss += span_loss
                avg_gate_loss += gate_loss
            model.zero_grad()
            # evaluation
            if steps != 0 and steps % logging_step == 0:
                print("\n" + "=" * 10 + "evaluation" + "=" * 10)
                print("Epoch {}, Step {}".format(epoch, steps))
                with torch.no_grad():
                    eval_result = evaluate(model, eval_dataloader, tokenizer, is_test=False)
                    test_result = evaluate(model, test_dataloader, tokenizer, is_test=True)
                print("Eval Result:", eval_result)
                print("Test Result:", test_result)
                # save model on dev
                if eval_result['em'] + eval_result['f1'] > save_dev_em + save_dev_f1:
                    save_dev_f1 = eval_result['f1']
                    save_dev_em = eval_result['em']
                    torch.save(model.state_dict(), model_save_path)
                    result = {"epoch": epoch, "test": test_result, "dev": eval_result}
                    print("save model with only dev em: %f,f1: %f" % (save_dev_em, save_dev_f1))
                    print("test em: %f,f1: %f" % (test_result['em'], test_result['f1']))
                    with open(result_json_save_path, "w") as f:
                        json.dump(result, f)

            steps += 1
        print(
            "\nAverage Loss:%.3f | GateLoss:%.3f | SpanLoss:%.3f | UtterLoss:%.3f | SpeakerLoss:%.3f " \
            % (avg_loss / len(train_loader), avg_gate_loss / len(train_loader), avg_span_loss / len(train_loader),
               avg_utter_loss / len(train_loader), avg_speaker_loss / len(train_loader)))


def evaluate(model, eval_loader, tokenizer, is_test=False, with_na=True):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    answer_dict, na_dict = {}, {}
    correct_num, all_num = 0, 0

    for batch in eval_loader:
        cur_batch_size = len(batch['input_ids'])

        inputs = {'input_ids': batch['input_ids'],
                  'token_type_ids': batch['token_type_ids'],
                  'attention_mask': batch['attention_mask'],
                  'p_mask': batch['p_mask'],
                  'context': batch['context'],
                  'utterance_ids_dict': batch['utterance_ids_dict'],
                  'offset_mapping': batch['offset_mapping'],
                  'qid': batch['qid'],
                  }
        if args.add_speaker_mask:
            inputs.update({'speaker_ids_dict': batch['speaker_ids_dict']})
        outputs = model(**inputs)
        answer_list, na_list = outputs[0], outputs[1]
        if args.add_speaker_mask:
            b_correct_num, b_all_num = outputs[2]
            correct_num += b_correct_num
            all_num += b_all_num
        for qid, ans_text in answer_list:
            answer_dict[qid] = ans_text
        for qid, na_prob in na_list:
            na_dict[qid] = na_prob
    with open(args.pred_file, "w") as f:
        json.dump(answer_dict, f, indent=2)
    with open(args.na_file, "w") as f:
        json.dump(na_dict, f, indent=2)
    if args.add_speaker_mask:
        print("Speaker prediction acc: %.5f" % (correct_num / all_num))
    evaluate_options = EVAL_OPTS(data_file=test_path if is_test else eval_path,
                                 pred_file=args.pred_file,
                                 na_prob_file=args.na_file if with_na else None)
    res = evaluate_on_squad(evaluate_options)
    em = res['exact']
    f1 = res['f1']
    rtv_dict = {'em': em, 'f1': f1, 'HasAns_exact': res['HasAns_exact'], 'HasAns_f1': res['HasAns_f1'],
                'NoAns_exact': res['NoAns_exact'], 'NoAns_f1': res['NoAns_f1']}
    model.train()

    return rtv_dict


if __name__ == "__main__":
    print(args.model_type)
    set_seed()
    MRCModel = MRC_MODEL_LIST[args.model_num].MRCModel
    print("model:", MRC_MODEL_LIST[args.model_num])

    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    config = config_class.from_pretrained(args.model_name)
    if args.model_type != 'xlnet':
        config.start_n_top = 5
        config.end_n_top = 5

    # create datasets
    train_dataset = get_dataset(train_path, args.cache_path, \
                                tokenizer, args.max_length, training=True)
    eval_dataset = get_dataset(eval_path, args.cache_path, \
                               tokenizer, args.max_length, training=False)
    test_dataset = get_dataset(test_path, args.cache_path, \
                               tokenizer, args.max_length, training=False)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=collate_fn)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MRCModel.from_pretrained(args.model_name, config=config)

    if hasattr(model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.load_mha_params()

    if os.path.exists(pretrained_model_path):
        # load checkpoint on SQuAD
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        print("=" * 10 + "load pretrained model" + "=" * 10)
        model = model.to(device)
        train(model, train_dataloader, eval_dataloader, test_dataloader, tokenizer)

    else:
        print("Please run train_squad first.")

    model = model.to(device)
    load_checkpoint = True
    if load_checkpoint:
        model.load_state_dict(torch.load(model_save_path))
        print("=" * 10 + "load checkpoint" + "=" * 10)
        # eval
        print("=" * 10 + "start evaluation" + "=" * 10)
        with torch.no_grad():
            eval_result = evaluate(model, eval_dataloader, tokenizer, is_test=False)
            test_result = evaluate(model, test_dataloader, tokenizer, is_test=True)
        print("Eval Result:", eval_result)
        print("Test Result:", test_result)