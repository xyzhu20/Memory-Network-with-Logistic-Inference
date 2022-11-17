import json
import torch
import numpy as np
import random
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, XLNetTokenizerFast, ElectraTokenizerFast
from transformers import BertConfig, XLNetConfig, ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.squad_evaluate import EVAL_OPTS_SQuAD, main as evaluate_on_squadv2

from utils.config import *
from models.ours import LIMN

from utils.utils_split import collate_fn
from utils.squad_utils import get_squad_dataset

MRC_MODEL_LIST = [None, None, LIMN]

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizerFast),
    'xlnet': (XLNetConfig, XLNetTokenizerFast),
    'electra': (ElectraConfig, ElectraTokenizerFast),
}

warnings.filterwarnings("ignore")
device = torch.device("cuda:" + str(args.cuda)) if USE_CUDA else torch.device("cpu")
config_class, tokenizer_class = MODEL_CLASSES[args.model_type]

model_save_path = "./saves/checkpoint/" + args.model_type + "_SQuAD_" + str(
    args.learning_rate) + "_T" + str(args.ts_num) + '.pkl'

result_json_save_path = "./saves/result/" + args.model_type + "_SQuAD_" + str(
    args.learning_rate) + "_T" + str(args.ts_num) + '.json'

def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def train_squad(model, train_loader, eval_dataloader, tokenizer, dev_f1=0, dev_em=0):
    print("Traning arguments:")
    print(args)

    patience_turns = 0
    model.train()
    model.zero_grad()

    # freeze some params not forward
    freeze = ['internal_fusion', 'QA_fusion', 'memory_fusion', 'QA_coattn', 'speaker_detector', 'utter_filter',
              'attn_fct', 'MHA_']

    for name, param in model.named_parameters():
        if any(nf in name for nf in freeze):
            param.requires_grad = False
            # print(name, param.requires_grad)

    # group params
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    #    optimizer_grouped_parameters = [
    #    {'params': [p for n, p in model.named_parameters() if
    #                (not any(nd in n for nd in no_decay)) and p.requires_grad == True],
    #     'weight_decay': args.weight_decay},
    #    {'params': [p for n, p in model.named_parameters() if
    #                (any(nd in n for nd in no_decay)) and p.requires_grad == True], 'weight_decay': 0.0}
    #    ]

    all_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_loader) * args.epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(all_optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    logging_step = t_total // (args.epochs * 5)
    steps = 0

    for epoch in range(args.epochs):
        avg_loss, avg_gate_loss, avg_span_loss, avg_utter_loss, avg_speaker_loss, avg_contrastive_loss = 0, 0, 0, 0, 0, 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, batch in pbar:
            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask'],
                      'p_mask': batch['p_mask'],
                      'start_pos': batch['start_pos'],
                      'end_pos': batch['end_pos'],
                      'is_impossible': batch['is_impossible'],
                      }

            outputs = model.train_MemEncoder(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            all_optimizer.step()
            if t_total is not None:
                scheduler.step()
            if len(outputs) == 3:
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
                    eval_result = evaluate_squad(model, eval_dataloader, tokenizer, is_test=False)
                print("Eval Result:", eval_result)

                # save model on dev
                if eval_result['em'] + eval_result['f1'] > dev_em + dev_f1:
                    dev_em = eval_result['em']
                    dev_f1 = eval_result['f1']
                    torch.save(model.state_dict(), model_save_path)
                    print("save model with dev em: %f,f1: %f" % (dev_em, dev_f1))
                    # save eval result
                    eval_path = result_json_save_path
                    result = {"epoch": epoch, "dev": eval_result}
                    with open(eval_path, "w") as f:
                        json.dump(result, f)
            steps += 1
        print(
            "\nAverage Loss:%.3f | GateLoss:%.3f | SpanLoss:%.3f | UtterLoss:%.3f | SpeakerLoss:%.3f | CTLoss:%.3f " \
            % (avg_loss / len(train_loader), avg_gate_loss / len(train_loader), avg_span_loss / len(train_loader),
               avg_utter_loss / len(train_loader), avg_speaker_loss / len(train_loader),
               avg_contrastive_loss / len(train_loader)))

    # load the best model and evaluate
    model.load_state_dict(torch.load(model_save_path))
    print("=" * 10 + "load best saves evaluation" + "=" * 10)
    with torch.no_grad():
        eval_result = evaluate_squad(model, eval_dataloader, tokenizer, is_test=False)
    print("Eval Result:", eval_result)


def evaluate_squad(model, eval_loader, tokenizer, is_test=False, with_na=True):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model.eval()
    answer_dict, na_dict = {}, {}

    for batch in eval_loader:
        cur_batch_size = len(batch['input_ids'])

        inputs = {'input_ids': batch['input_ids'],
                  'token_type_ids': batch['token_type_ids'],
                  'attention_mask': batch['attention_mask'],
                  'p_mask': batch['p_mask'],
                  'context': batch['context'],
                  'offset_mapping': batch['offset_mapping'],
                  'qid': batch['qid'],
                  }
        outputs = model.train_MemEncoder(**inputs)
        answer_list, na_list = outputs[0], outputs[1]
        for qid, ans_text in answer_list:
            answer_dict[qid] = ans_text
        for qid, na_prob in na_list:
            na_dict[qid] = na_prob
    with open(args.squad_pred_file, "w") as f:
        json.dump(answer_dict, f, indent=2)
    with open(args.squad_na_file, "w") as f:
        json.dump(na_dict, f, indent=2)
    evaluate_options = EVAL_OPTS_SQuAD(data_file=test_path if is_test else eval_path,
                                       pred_file=args.squad_pred_file,
                                       na_prob_file=args.squad_na_file if with_na else None)
    res = evaluate_on_squadv2(evaluate_options)
    em = res['exact']
    f1 = res['f1']
    rtv_dict = {'em': em, 'f1': f1, 'HasAns_exact': res['HasAns_exact'], 'HasAns_f1': res['HasAns_f1'],
                'NoAns_exact': res['NoAns_exact'], 'NoAns_f1': res['NoAns_f1']}
    model.train()

    return rtv_dict


if __name__ == "__main__":
    args.mha_layer_num = 0  # during pre-training
    print(args.model_type)
    print("Pre-training on SQuAD 2.0")
    set_seed()
    MRCModel = MRC_MODEL_LIST[args.model_num].MRCModel
    print("model:", MRC_MODEL_LIST[args.model_num])

    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    config = config_class.from_pretrained(args.model_name)
    if args.model_type != 'xlnet':
        config.start_n_top = 5
        config.end_n_top = 5

    train_path = "./data/squad2.0/train-v2.0.json"
    eval_path = "./data/squad2.0/dev-v2.0.json"
    # training
    train_dataset = get_squad_dataset(train_path, args.cache_path, \
                                      tokenizer, args.max_length, training=True)
    eval_dataset = get_squad_dataset(eval_path, args.cache_path, \
                                     tokenizer, args.max_length, training=False)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=collate_fn)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

    model = MRCModel.from_pretrained(args.model_name, config=config)
    if hasattr(model, 'load_mha_params'):
        print("Loading multi-head attention parameters from pretrained model...")
        model.load_mha_params()
    model = model.to(device)
    #train_squad(model, train_dataloader, eval_dataloader, tokenizer)

    model = model.to(device)
    load_checkpoint = True
    if load_checkpoint:
        model.load_state_dict(torch.load(model_save_path))
        print("=" * 10 + "load checkpoint" + "=" * 10)
        # eval
        print("=" * 10 + "start evaluation" + "=" * 10)
        with torch.no_grad():
            eval_result = evaluate_squad(model, eval_dataloader, tokenizer, is_test=False)
