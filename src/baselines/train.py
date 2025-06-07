import codecs
import numpy as np
import pandas as pd
import re
import math
import numpy as np
import random
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from transformers import AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup

from models import *

parser = argparse.ArgumentParser("Baselines")

parser.add_argument("--model_type", type=str, help="LogReg/RNN/HRED/BERT/GPT-2/DialoGPT/RoBERTa")
parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--lambda_EI", default=0.5, type=float, help="lambda_identification")
parser.add_argument("--lambda_RE", default=0.5, type=float, help="lambda_rationale")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
parser.add_argument("--max_len", default=64, type=int, help="maximum sequence length")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=4, type=int, help="number of epochs")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
parser.add_argument("--train_path", type=str, help="path to input training data")
parser.add_argument("--dev_path", type=str, help="path to input validation data")
parser.add_argument("--test_path", type=str, help="path to input test data")
parser.add_argument("--do_validation", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_model_path", type=str, help="path to save model")

args = parser.parse_args()

print("=====================Args====================")

print('baseline model = ', args.model_type)
print('lr = ', args.lr)
print('lambda_EI = ', args.lambda_EI)
print('lambda_RE = ', args.lambda_RE)
print('dropout = ', args.dropout)
print('max_len = ', args.max_len)
print('batch_size = ', args.batch_size)
print('epochs = ', args.epochs)
print('seed_val = ', args.seed_val)
print('train_path = ', args.train_path)
print('do_validation = ', args.do_validation)
print('do_test = ', args.do_test)

print("=============================================")

'''
Use GPU if available
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

'''
Load input dataset
'''
if args.train_path:
    df = pd.read_csv(args.train_path, delimiter=',')
    df['rationale_labels'] = df['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
else:
    print('No input training data specified.')
    print('Exiting...')
    exit(-1)

if args.do_test:
    if args.test_path:
        df_test = pd.read_csv(args.test_path, delimiter=',')
        df_test['rationale_labels'] = df_test['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
    else:
        print('No input test data specified.')
        print('Exiting...')
        exit(-1)

if args.do_validation:
    if args.dev_path:
        df_val = pd.read_csv(args.dev_path, delimiter=',')
        df_val['rationale_labels'] = df_val['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
    else:
        print('No input validation data specified.')
        print('Exiting...')
        exit(-1)

'''
Tokenize input
'''
if args.model_type == 'LogReg':
    texts = (df["seeker_post"].fillna("") + " [SEP] " + df["response_post"].fillna("")).to_list()
    labels = df.level.values.astype(int)
    labels = torch.tensor(labels)
    rationales = df.rationale_labels.values.tolist()
    rationales = torch.stack(rationales, dim=0)
    vectorizer = LogReg.vectorizer(None)
    x_train = vectorizer.fit_transform(texts)

    if args.do_validation:
        val_texts = (df_val["seeker_post"].fillna("") + " [SEP] " + df_val["response_post"].fillna("")).to_list()
        val_labels = df_val.level.values.astype(int)
        val_labels = torch.tensor(val_labels)
        val_rationales = df_val.rationale_labels.values.tolist()
        val_rationales = torch.stack(val_rationales, dim=0)
        x_val = vectorizer.transform(val_texts)

    if args.do_test:
        test_texts = (df_test["seeker_post"].fillna("") + " [SEP] " + df_test["response_post"].fillna("")).to_list()
        test_labels = df_test.level.values.astype(int)
        test_labels = torch.tensor(test_labels)
        test_rationales = df_test.rationale_labels.values.tolist()
        test_rationales = torch.stack(test_rationales, dim=0)
        x_test = vectorizer.transform(test_texts)

elif args.model_type == 'RNN':
    train_vocab = RNNDataset(df)
    vocab_size = train_vocab.vocab_size
    word2idx = train_vocab.word2idx

    if args.do_validation:
        val_vocab = RNNDataset(df_val, word2idx=word2idx)
    if args.do_test:
        test_vocab = RNNDataset(df_test, word2idx=word2idx)

elif args.model_type == 'HRED':
    pass

elif args.model_type in ('BERT', 'GPT-2', 'DialoGPT', 'RoBERTa'):
    if args.model_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif args.model_type == 'GPT-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model_type == 'DialoGPT':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small', do_lower_case=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model_type == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    train_tokenizer = tokenizer.batch_encode_plus((df["seeker_post"].fillna("") + " [SEP] " + df["response_post"].fillna("")).to_list(), add_special_tokens=True, max_length=args.max_len, pad_to_max_length=True, return_attention_mask=True)
    input_ids = torch.tensor(train_tokenizer['input_ids'])
    attention_masks = torch.tensor(train_tokenizer['attention_mask'])

    labels = df.level.values.astype(int)
    labels = torch.tensor(labels)
    rationales = df.rationale_labels.values.tolist()
    rationales = torch.stack(rationales, dim=0)

    if args.do_validation:
        val_tokenizer = tokenizer.batch_encode_plus((df_val["seeker_post"].fillna("") + " [SEP] " + df_val["response_post"].fillna("")).to_list(), add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_mask=True)
        val_input_ids = torch.tensor(val_tokenizer['input_ids'])
        val_attention_masks = torch.tensor(val_tokenizer['attention_mask'])

        val_labels = torch.tensor(df_val.level.values.astype(int))
        val_rationales = df_val.rationale_labels.values.tolist()
        val_rationales = torch.stack(val_rationales, dim=0)
        val_rationales_trimmed = torch.tensor(df_val.rationale_labels_trimmed.values.astype(int))

    if args.do_test:
        test_tokenizer = tokenizer.batch_encode_plus((df_test["seeker_post"].fillna("") + " [SEP] " + df_test["response_post"].fillna("")).to_list(), add_special_tokens=True,max_length=args.max_len, pad_to_max_length=True, return_attention_mask=True)
        test_input_ids = torch.tensor(test_tokenizer['input_ids'])
        test_attention_masks = torch.tensor(test_tokenizer['attention_mask'])

        test_labels = torch.tensor(df_test.level.values.astype(int))
        test_rationales = df_test.rationale_labels.values.tolist()
        test_rationales = torch.stack(test_rationales, dim=0)
        test_rationales_trimmed = torch.tensor(df_test.rationale_labels_trimmed.values.astype(int))

else:
    print('No baseline model type specified.')
    print('Exiting...')
    exit(-1)

'''
Load model
'''
if args.model_type == 'LogReg':
    model = LogReg()
elif args.model_type == 'RNN':
    model = TwoLayerRNN(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, output_dim=3, n_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
elif args.model_type == 'HRED':
    pass
elif args.model_type == 'BERT':
    model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(df.level.unique())
        )
    optimizer = AdamW(model.parameters(),
                lr = args.lr,
                eps = 1e-8
                )
    model = model.to(device)
elif args.model_type == 'GPT-2':
    model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2',
            num_labels=len(df.level.unique())
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = AdamW(model.parameters(),
                lr = args.lr,
                eps = 1e-8
                )
    model = model.to(device)
elif args.model_type == 'DialoGPT':
    model = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/DialoGPT-small',
            num_labels=len(df.level.unique())
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = AdamW(model.parameters(),
                lr = args.lr,
                eps = 1e-8
                )
    model = model.to(device)
elif args.model_type == 'RoBERTa':
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(df.level.unique())
    )
    optimizer = AdamW(model.parameters(),
                lr = args.lr,
                eps = 1e-8
                )
    model = model.to(device)
else:
    print('No baseline model type specified.')
    print('Exiting...')
    exit(-1)

'''
Training schedule
'''
if args.model_type == 'LogReg':
    pass
elif args.model_type == 'RNN':
    # train_dataset = TensorDataset(train_vocab, labels, rationales)
    # train_size = int(len(train_dataset))
    # train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = args.batch_size)
    train_size = int(len(train_vocab))
    train_dataloader = DataLoader(train_vocab, batch_size=args.batch_size, shuffle=True, collate_fn=RNNDataset.collate_rnn)
    if args.do_validation:
        # val_dataset = TensorDataset(x_val, val_labels, val_rationales)
        # val_dataloader = DataLoader(val_dataset, sampler = RandomSampler(val_dataset), batch_size = args.batch_size)
        val_dataloader = DataLoader(val_vocab, batch_size=args.batch_size, shuffle=False, collate_fn=RNNDataset.collate_rnn)
    if args.do_test:
        # test_dataset = TensorDataset(x_test, test_labels, test_rationales)
        # test_dataloader = DataLoader(test_dataset, sampler = RandomSampler(test_dataset), batch_size = args.batch_size)
        test_dataloader = DataLoader(test_vocab, batch_size=args.batch_size, shuffle=False, collate_fn=RNNDataset.collate_rnn)

    total_steps = len(train_dataloader) * args.epochs
    num_batch = len(train_dataloader)

    print('total_steps =', total_steps)
    print('num_batch =', num_batch)
    print("=============================================")
elif args.model_type == 'HRED':
    pass
elif args.model_type in ('BERT', 'GPT-2', 'DialoGPT', 'RoBERTa'):
    train_dataset = TensorDataset(input_ids, attention_masks, labels, rationales)
    train_size = int(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = args.batch_size)

    if args.do_validation:
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels, val_rationales, val_rationales_trimmed)
        val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = args.batch_size)

    if args.do_test:
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels, test_rationales, test_rationales_trimmed)
        test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = args.batch_size)


    total_steps = len(train_dataloader) * args.epochs
    num_batch = len(train_dataloader)

    print('total_steps =', total_steps)
    print('num_batch =', num_batch)
    print("=============================================")

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
else:
    print('No baseline model type specified.')
    print('Exiting...')
    exit(-1)

random.seed(args.seed_val)
np.random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
torch.cuda.manual_seed_all(args.seed_val)

'''
Do Training
'''
if args.model_type == 'LogReg':
    model.fit(x_train, labels)
    if args.do_validation:
        print('\n\nRunning validation...\n')
        preds, flat_acc, f1 = model.predict(x_val, val_labels, val_rationales)
        print("  Accuracy-Empathy: {0:.4f}".format(flat_acc))
        print("  macro_f1_empathy: {0:.4f}".format(f1))

        # print("  Accuracy-Rationale: {0:.4f}".format(flat_acc_rationale))
        # print("  IOU-F1-Rationale: {0:.4f}".format(iou_f1_rationale))
        # print("  macro_f1_rationale: {0:.4f}".format(macro_f1_rationale))

        print('\n')
else:
    for epoch_i in range(0, args.epochs):
        total_train_loss = 0
        total_train_empathy_loss = 0
        total_train_rationale_loss = 0

        pbar = tqdm(total=num_batch, desc=f"training")

        model.train()

        for step, batch in enumerate(train_dataloader):
            if args.model_type == 'RNN':
                idxs = batch[0].to(device)
                lengths = batch[1].to(device)
                labels = batch[2].to(device)

                optimizer.zero_grad()
                logits = model(idxs, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            elif args.model_type == 'HRED':
                pass
            elif args.model_type in ('BERT', 'GPT-2', 'DialoGPT', 'RoBERTa'):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_rationales = batch[3].to(device)

                optimizer.zero_grad()
                outputs = model(
                    input_ids = b_input_ids,
                    attention_mask = b_input_mask,
                    labels = b_labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix_str(
                    f"total loss: {float(total_train_loss/(step+1)):.4f} epoch: {epoch_i}")
                pbar.update(1)
                total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_empathy_loss = total_train_empathy_loss / len(train_dataloader)
        avg_train_rationale_loss = total_train_rationale_loss / len(train_dataloader)

        pbar.close()

        if args.do_validation:
            '''
            Validation
            '''
            print('\n\nRunning validation...\n')
            model.eval()
            total_eval_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_dataloader:
                    if args.model_type == 'RNN':
                        idxs = batch[0].to(device)
                        lengths = batch[1].to(device)
                        labels = batch[2].to(device)
                        logits = model(idxs, lengths)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        all_preds.extend(preds.tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                    elif args.model_type == 'HRED':
                        pass
                    elif args.model_type in ('BERT', 'GPT-2', 'DialoGPT', 'RoBERTa'):
                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)
                        b_rationales = batch[3].to(device)
                        outputs = model(
                            input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            labels = b_labels
                        )
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        all_preds.extend(preds.tolist())
                        all_labels.extend(b_labels.cpu().numpy().tolist())

            flat_acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            print("  Accuracy-Empathy: {0:.4f}".format(flat_acc))
            print("  macro_f1_empathy: {0:.4f}".format(f1))

'''
Test
'''
if args.do_test:
    print("\n\nRunning test...\n")
    if args.model_type == 'LogReg':
        preds, flat_acc, f1 = model.predict(x_test, test_labels, test_rationales)
        print("  Accuracy-Empathy: {0:.4f}".format(flat_acc))
        print("  macro_f1_empathy: {0:.4f}".format(f1))

        # print("  Accuracy-Rationale: {0:.4f}".format(flat_acc_rationale))
        # print("  IOU-F1-Rationale: {0:.4f}".format(iou_f1_rationale))
        # print("  macro_f1_rationale: {0:.4f}".format(macro_f1_rationale))

        print('\n')
    else:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                if args.model_type == 'RNN':
                    idxs = batch[0].to(device)
                    lengths = batch[1].to(device)
                    labels = batch[2].to(device)
                    logits = model(idxs, lengths)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                elif args.model_type == 'HRED':
                    pass
                elif args.model_type in ('BERT', 'GPT-2', 'DialoGPT', 'RoBERTa'):
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    b_rationales = batch[3].to(device)
                    outputs = model(
                        input_ids = b_input_ids,
                        attention_mask = b_input_mask,
                        labels = b_labels
                    )
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(b_labels.cpu().numpy().tolist())
        flat_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        print("  Accuracy-Empathy: {0:.4f}".format(flat_acc))
        print("  macro_f1_empathy: {0:.4f}".format(f1))

if args.save_model:
    if args.model_type == 'LogReg':
        import pickle
        with open(args.save_model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        torch.save(model.state_dict(), args.save_model_path)