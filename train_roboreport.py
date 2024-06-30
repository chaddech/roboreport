import torch
from transformers import TrainingArguments
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from datasets import load_metric
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import IPython
import gc
import pickle
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from shutil import copyfile
import sys
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from collections import Counter, defaultdict
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer
import copy
from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
import torch.multiprocessing
from roboreport_helper_files.roboreport_helper_methods import *
from roboreport_helper_files.roboreport_metrics_methods import *
from roboreport_helper_files.roboreport_dataset import DictMultitaskDataset
from roboreport_helper_files.roboreport_config_file import args
from roboreport_helper_files.generate_questions import make_nat_lang_right_before_questions, make_nat_lang_right_after_questions

helper_files = ['roboreport_helper_methods.py', 'roboreport_metrics_methods.py', 
    'roboreport_dataset.py', 'roboreport_config_file.py', 'generate_questions.py']

start_time = time.time()

OUTPUT_INFO_FOR_NAME =   args.output_info_for_dir_name + str(args.batch_size) + '_' + str(args.num_workers)+'workers_seed' + str(args.seed) + '_lr' +  str(args.learning_rate)+'_'  + '_cuda' + str(args.device_num)+"_"+os.uname()[1] 
output_directory = 'roboreport_results/' + OUTPUT_INFO_FOR_NAME + '_' + str(start_time) + "/"

transformer_model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)

if args.full_dataset:
    allowed_subset_of_train_ids = None
    allowed_subset_of_valid_seen_ids = None
    allowed_subset_of_valid_unseen_ids = None
    print("using full dataset")


f = open(VALID_SEEN_ACTION_COUNTER_FILE, 'rb')
valid_seen_action_counter = pickle.load(f)
f.close()


f = open(VALID_UNSEEN_ACTION_COUNTER_FILE, 'rb')
valid_unseen_action_counter = pickle.load(f)
f.close()

f = open(TRAIN_ACTION_COUNTER_FILE, 'rb')
train_action_counter = pickle.load(f)
f.close()


best_val_acc = 0
unseen_val_acc_at_best_seen_val_epoch = 0
best_unseen_val_acc = 0
best_unseen_val_epoch = 0

qa_task_descriptions = {}
main_qa_task_descriptions = {}
secondary_task_descriptions = {}

qa_task_descriptions['pddl_summ'] = {'name': 'pddl_summ', 
    'text_input' : 'summarize_instruction', 
    'target_output': 'clean_high_pddl', 'summarize_instruction': 'narrate what you did.'}

main_qa_task_descriptions['pddl_summ'] = {'name': 'pddl_summ', 
    'text_input' : 'summarize_instruction', 
    'target_output': 'clean_high_pddl', 'summarize_instruction': 'narrate what you did.'}



qa_task_descriptions['nat_lang_summ'] = {'name': 'nat_lang_summ', 
    'text_input' : 'nat_lang_summarize_instruction', 
    'target_output': 'task_desc_summaries', 'nat_lang_summarize_instruction': 'summarize what you did.'}

main_qa_task_descriptions['nat_lang_summ'] = {'name': 'nat_lang_summ', 
    'text_input' : 'nat_lang_summarize_instruction', 
    'target_output': 'task_desc_summaries', 'nat_lang_summarize_instruction': 'summarize what you did.'}


qa_task_descriptions['right_before_questions'] = {'name': 'right_before_questions', 
    'text_input' : 'right_before_input', 
    'target_output': 'right_before_output'}

qa_task_descriptions['right_after_questions'] = {'name': 'right_after_questions', 
    'text_input' : 'right_after_input', 
    'target_output': 'right_after_output'}


secondary_task_descriptions['right_before_questions'] = {'name': 'right_before_questions', 
    'text_input' : 'right_before_input', 
    'target_output': 'right_before_output'}

secondary_task_descriptions['right_after_questions'] = {'name': 'right_after_questions', 
    'text_input' : 'right_after_input', 
    'target_output': 'right_after_output'}


qa_task_descriptions['nat_lang_right_before_questions'] = {'name': 'nat_lang_right_before_questions', 
    'text_input' : 'right_before_input', 
    'target_output': 'right_before_output'}

secondary_task_descriptions['nat_lang_right_before_questions'] = {'name': 'nat_lang_right_before_questions', 
    'text_input' : 'right_after_input', 
    'target_output': 'right_after_output'}


qa_task_descriptions['nat_lang_right_after_questions'] = {'name': 'nat_lang_right_after_questions', 
    'text_input' : 'right_before_input', 
    'target_output': 'right_before_output'}

secondary_task_descriptions['nat_lang_right_after_questions'] = {'name': 'nat_lang_right_after_questions', 
    'text_input' : 'right_after_input', 
    'target_output': 'right_after_output'}



secondary_task_descriptions['object_either_or'] = {'name': 'object_either_or', 
    'text_input' : 'simple_object_either_or_input', 
    'target_output': 'simple_object_either_or_output'}


qa_task_descriptions['object_either_or'] = {'name': 'object_either_or', 
    'text_input' : 'simple_object_either_or_input', 
    'target_output': 'simple_object_either_or_output'}


secondary_task_descriptions['nat_lang_action_yes_no'] = {'name': 'nat_lang_action_yes_no', 
    'text_input' : 'nat_lang_action_yes_no_input', 
    'target_output': 'nat_lang_action_yes_no_output'}


qa_task_descriptions['nat_lang_action_yes_no'] = {'name': 'nat_lang_action_yes_no', 
    'text_input' : 'nat_lang_action_yes_no_input', 
    'target_output': 'nat_lang_action_yes_no_input'}


secondary_task_descriptions['action_either_or'] = {'name': 'action_either_or', 
    'text_input' : 'action_either_or_input', 
    'target_output': 'action_either_or_output'}


qa_task_descriptions['action_either_or'] = {'name': 'action_either_or', 
    'text_input' : 'action_either_or_input', 
    'target_output': 'action_either_or_output'}


secondary_task_descriptions['simple_object_yes_no'] = {'name': 'simple_object_yes_no', 
    'text_input' : 'simple_object_qa_input', 
    'target_output': 'simple_object_qa_answer'}

qa_task_descriptions['simple_object_yes_no'] = {'name': 'simple_object_yes_no', 
'text_input' : 'simple_object_qa_input', 
'target_output': 'simple_object_qa_answer'}



secondary_task_descriptions['simple_action_yes_no'] = {'name': 'simple_action_yes_no', 
    'text_input' : 'simple_action_qa_input', 
    'target_output': 'simple_action_qa_answer'}

qa_task_descriptions['simple_action_yes_no'] = {'name': 'simple_action_yes_no', 
'text_input' : 'simple_action_qa_input', 
'target_output': 'simple_action_qa_answer'}

if args.static_validate:
    print("static validate")
    static_validate_file = "10_action_indices_per_episode_act_for_5_steps_for_static_evaluation.pkl"
    print(static_validate_file)
    f=open(static_validate_file, 'rb')
    static_validate_dicts = pickle.load(f)
    f.close()
    output_directory = '../static_validate_results/' + saved_model_state_dict.split('/')[-2]+args.main_task+'/'



if not os.path.exists(output_directory):
    os.mkdir(output_directory)


copyfile(sys.argv[0], output_directory +  sys.argv[0])
for helper_file in helper_files:
    copyfile('roboreport_helper_files/'+helper_file, output_directory +  helper_file)


writer = SummaryWriter(output_directory)

fuller_metrics_to_save = []
eval_preds_to_save = []
all_record_dicts = []
all_metrics_dicts = []
full_epoch_accuracy_dicts = []
logs_to_save = []
global_epoch = 0
best_val_epoch = 0
best_val_seen_rouge_f1 = 0

def set_seed_everywhere(seed):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(seed)
    random.seed(args.seed)


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.input_feature_size = 1568


        if args.use_clip_feats:
            self.num_initial_layers = 2048
        else:

            self.num_initial_layers = 512

        self.size_initial_layer = 7

        if args.use_clip_feats:
            self.conv1 = nn.Conv2d(2048,1024,1)
            self.conv2 = nn.Conv2d(1024,128,1)
            self.conv3 = nn.Conv2d(128, 32, 1)

            self.bn1 = nn.BatchNorm2d(1024)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(32)

        else:

            self.conv1 = nn.Conv2d(512,256,1)
            self.bn1 = nn.BatchNorm2d(256)


            self.conv2 = nn.Conv2d(256, 64, 1)

            self.bn2 = nn.BatchNorm2d(64)


            self.conv3 = nn.Conv2d(64, 32, 1)
            self.bn3 = nn.BatchNorm2d(32)


        self.ff = nn.Linear(1568, args.model_embed_size)



    def forward(self, x_source):
        xs_after_conv = []


        for input_x in x_source:
            input_x = input_x.to(args.device)

            input_x = input_x.reshape((input_x.shape[0], self.num_initial_layers, self.size_initial_layer, self.size_initial_layer))
            #output_x = F.relu(self.bn0(input_x))

            output_x = F.relu(self.conv1(input_x))
            output_x = self.bn1(output_x)
            output_x = F.relu(self.conv2(output_x))
            output_x = self.bn2(output_x)

            output_x = F.relu(self.conv3(output_x))
            output_x = self.bn3(output_x)

            output_x = output_x.reshape((input_x.shape[0], self.input_feature_size))

            output_x = F.relu(self.ff(output_x))

            xs_after_conv.append(output_x)
        
        return xs_after_conv



class SummModel(nn.Module):
    def __init__(self, transformer_model):

        super(SummModel, self).__init__()


        self.transformer_model = transformer_model
        self.transformer_model.config.max_length = args.max_length
        self.transformer_model.config.max_position_embeddings = args.max_length
        self.transformer_model = self.transformer_model.to(args.device)
        self.pad_token = self.transformer_model.encoder.embed_tokens(torch.tensor((0)).to(args.device))
        self.cnn_encoder = CNNEncoder()

        self.cnn_encoder = self.cnn_encoder.to(args.device)

        self.tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, model_max_length=args.max_length)
        if args.use_trad_clip_feats:

            self.trad_clip_feat_foward_layer = nn.Linear(1024, args.model_embed_size)

            self.trad_clip_feat_foward_layer = self.trad_clip_feat_foward_layer.to(args.device)


    def forward(self, image_tensors, source_text, embgram_tensors= None, target_sequence=None, in_training_phase = True):

        if len(image_tensors) > 0:
            if args.use_trad_clip_feats:
                conved_inputs = []
                for episode_tensors in image_tensors:
                    episode_tensors = episode_tensors.float()
                    episode_tensors = episode_tensors.to(args.device)
                    episode_tensors = self.trad_clip_feat_foward_layer(episode_tensors)
                    #episode_tensors = self.trad_clip_feat_bn(episode_tensors)
                    conved_inputs.append(episode_tensors)
            else:
                conved_inputs = self.cnn_encoder(image_tensors)


        if len(source_text) > 0 and source_text[0] != None:
            source_ids = self.tokenizer(source_text).input_ids
            

            source_embeds = []

            for z in source_ids:
                z = torch.tensor(z).to(args.device)
                embeds = self.transformer_model.encoder.embed_tokens(z)
                embeds = embeds.to(args.device)
                source_embeds.append(embeds)

            concat_inputs = []



            for x,y in zip(source_embeds, conved_inputs):
                y = y.to(args.device)
                images_prefix = ' images:'

                images_prefix = self.tokenizer(images_prefix).input_ids

                images_prefix_embeds = self.transformer_model.encoder.embed_tokens(torch.tensor(images_prefix[:-1]).to(args.device))
                images_suffix_embeds = self.transformer_model.encoder.embed_tokens(torch.tensor(images_prefix[-1]).to(args.device))

                tensor_to_append = torch.cat((x, images_prefix_embeds), dim=0)


                tensor_to_append = torch.cat((tensor_to_append, y), dim=0)

                images_suffix_embeds = torch.unsqueeze(images_suffix_embeds, 0)
                tensor_to_append = torch.cat((tensor_to_append, images_suffix_embeds), dim=0)
                concat_inputs.append(tensor_to_append)




        else:
            print('missing text inputs')
        try:

            sequences_padded = torch.nn.utils.rnn.pad_sequence(concat_inputs, batch_first=True)
        except:
            print("sequences_padded exception")
            IPython.embed()




        mask = torch.FloatTensor(torch.zeros((sequences_padded.shape[:2])))

        for x in range(len(concat_inputs)):
            sh = concat_inputs[x].shape[0] 
            mask[x][:sh] = 1




        mask = mask.to(args.device)

        sequences_padded = sequences_padded.to(args.device)
        max_len_this_batch = 0
        for x in target_sequence:
            if len(x) > max_len_this_batch:
                max_len_this_batch = len(x) + 5


        labels = self.tokenizer(target_sequence, padding='longest', max_length=args.max_length).input_ids


        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100

        labels = labels.to(args.device)


        if in_training_phase:

            to_return = self.transformer_model(inputs_embeds = sequences_padded, attention_mask=mask, labels=labels)
        
        else:
            to_return_forward = self.transformer_model(inputs_embeds = sequences_padded, attention_mask=mask, labels=labels)

            generated_to_return = self.transformer_model.generate(inputs_embeds = sequences_padded, attention_mask = mask)

            to_return = [generated_to_return, to_return_forward]

        return to_return



def save_records(comment=""):
    to_save=[args, all_record_dicts, all_metrics_dicts, train_state, logs_to_save, eval_preds_to_save, fuller_metrics_to_save, full_epoch_accuracy_dicts]
    f=open(output_directory + "log_and_examples.pkl", "wb")
    pickle.dump(to_save, f)
    f.close()

def save_general_checkpoint():

    global global_epoch
    global best_val_acc
    global best_unseen_val_acc
    global unseen_val_acc_at_best_seen_val_epoch
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'unseen_val_acc_at_best_seen_val_epoch': unseen_val_acc_at_best_seen_val_epoch,
            'best_unseen_val_acc': best_unseen_val_acc,
            'torch_rng_state': torch_rng_state,
            'np_rng_state': np_rng_state,
            }, output_directory + 'general_checkpoint.pt')



class VisFeatsPadSequence:
    def __call__(self, batch):

        return batch


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'valid_unseen_dataset_val_loss': [],
            'valid_unseen_dataset_val_acc': [],

            'train_dataset_val_loss': [],
            'train_dataset_val_acc': [],

            'valid_seen_dataset_val_loss': [],
            'valid_seen_dataset_val_acc': [],
            'valid_seen_dataset_val_index': [],
            'valid_unseen_dataset_val_index': [],
            'train_dataset_val_index': [],

            'rouge_scores': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}




def train(model, train_dataset, valid_unseen_datasets, valid_seen_datasets, train_state, dataloader,
    optimizer, scheduler, limit_tasks_list = None, validate_every=500, print_every=200, learning_rate=0.01):
    global logs_to_save
    #global current_eval_dataset
    global best_val_seen_rouge_f1
    global best_val_acc
    global best_unseen_val_acc
    global best_unseen_val_epoch
    global global_epoch
    global best_val_epoch
    global all_record_dicts
    global all_metrics_dicts
    global unseen_val_acc_at_best_seen_val_epoch

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    running_acc = 0



    data_len = train_dataset.__len__()


    counter = 0

    model.train()

    accuracy_stats = {}
    for task_name in train_dataset.qa_task_descriptions.keys():
        accuracy_stats[task_name] = Step_by_step_compute_accuracy_stats(task_name, train_dataset.dataset_name)


    for task_name in train_dataset.secondary_task_descriptions.keys():
        accuracy_stats[task_name] = Step_by_step_compute_accuracy_stats(task_name, train_dataset.dataset_name)


    running_losses = defaultdict(float)
    running_accs = defaultdict(float)

    train_epoch_outputs = defaultdict(list)
    train_epoch_labels = defaultdict(list)
    train_epoch_questions = defaultdict(list)
    all_episode_ids = defaultdict(list)

    optimizer.zero_grad()




    for i, batch_dicts in enumerate(dataloader):


        if i%15 == 0:
            gc.collect()


        this_batch_missing_words = []

        this_batch_acc_stat_dicts = {}
        for task_name, task_dict in train_dataset.main_qa_task_descriptions.items():




            embgram_tensors = []
            input_images_tensors = []
            target_tensors = []
            text_input = []
            episode_ids = []



            for this_dict in batch_dicts:


                if args.exclude_some_pddl:
                    if this_dict['episode_id'] in pddl_exclusions['train_ids_to_exclude']:
                        continue

                if task_name in this_dict:
                    if 'images' in this_dict:
                        input_images_tensors.append(this_dict['images'])
                    episode_ids.append(this_dict['episode_id'])
                    target_tensors.append(this_dict[task_name]['target_output'])
                    text_input.append(this_dict[task_name]['text_input'])



            if len(input_images_tensors) == 0:
                continue

            try:
                model_output = model(image_tensors=input_images_tensors, source_text=text_input, target_sequence = target_tensors)
            except Exception as this_exception:
                print(episode_ids)
                print(str(this_exception))
                IPython.embed()
                continue

            model_output['loss'].backward()

            if ((i+1) % args.num_batches_before_step == 0) or (i+1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()


            # get decoder to ignore everything after EOS token
            raw_outputs = model_output['logits'].argmax(dim=2)
            eos_indices = torch.where(raw_outputs == 1)

            for z in range(len(eos_indices[0])):
                raw_outputs[eos_indices[0][z]][eos_indices[1][z]:] = 1




            these_outputs = model.tokenizer.batch_decode(raw_outputs, 
                skip_special_tokens=True)

            for w in these_outputs:
                train_epoch_outputs[task_name].append(w)
            for w in target_tensors:
                train_epoch_labels[task_name].append(w)
            for w in text_input:
                train_epoch_questions[task_name].append(w)
            for ep_id in episode_ids:
                all_episode_ids[task_name].append(ep_id)



            loss = model_output['loss'].detach().item()



            running_losses[task_name] += loss

            del loss, model_output


            this_acc_dict = accuracy_stats[task_name].compute_step(text_input, target_tensors,
                these_outputs, episode_ids, global_epoch)


            this_batch_acc_stat_dicts[task_name] = this_acc_dict





        for task_name, task_dict in train_dataset.secondary_task_descriptions.items():




            embgram_tensors = []
            input_images_tensors = []
            target_tensors = []
            text_input = []
            episode_ids = []



            for this_dict in batch_dicts:


                if task_name in this_dict:
                    if 'images' in this_dict:
                        input_images_tensors.append(this_dict['images'])
                    episode_ids.append(this_dict['episode_id'])
                    target_tensors.append(this_dict[task_name]['target_output'])
                    text_input.append(this_dict[task_name]['text_input'])



            if len(input_images_tensors) == 0:
                continue

            try:
                model_output = model(image_tensors=input_images_tensors, source_text=text_input, target_sequence = target_tensors)
            except Exception as this_exception:
                print(episode_ids)
                print(str(this_exception))
                IPython.embed()
                continue

            model_output['loss'].backward()

            if ((i+1) % args.num_batches_before_step == 0) or (i+1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()


            # get decoder to ignore everything after EOS token
            raw_outputs = model_output['logits'].argmax(dim=2)
            eos_indices = torch.where(raw_outputs == 1)

            for z in range(len(eos_indices[0])):
                raw_outputs[eos_indices[0][z]][eos_indices[1][z]:] = 1




            these_outputs = model.tokenizer.batch_decode(raw_outputs, 
                skip_special_tokens=True)

            for w in these_outputs:
                train_epoch_outputs[task_name].append(w)
            for w in target_tensors:
                train_epoch_labels[task_name].append(w)
            for w in text_input:
                train_epoch_questions[task_name].append(w)
            for ep_id in episode_ids:
                all_episode_ids[task_name].append(ep_id)



            loss = model_output['loss'].detach().item()



            running_losses[task_name] += loss

            del loss, model_output


            this_acc_dict = accuracy_stats[task_name].compute_step(text_input, target_tensors,
                these_outputs, episode_ids, global_epoch)


            this_batch_acc_stat_dicts[task_name] = this_acc_dict




    num_minibatches = i + 1
    train_epoch_accuracy_stats_dicts = {}

    train_epoch_metrics_dicts = {}

    for k in train_epoch_questions.keys():

        train_epoch_dict = accuracy_stats[k].return_full_record_dict()
        train_epoch_dict['epoch'] = global_epoch
        train_epoch_accuracy_stats_dicts[k] = train_epoch_dict

        this_task_epoch_loss = running_losses[k]/num_minibatches

        writer.add_scalar('loss/'+k+'/train', this_task_epoch_loss, global_epoch)

        this_task_acc = train_epoch_dict['number_of_examples_correct']/train_epoch_dict['total_number_of_examples']
        writer.add_scalar('acc/'+k+'/train', this_task_acc, global_epoch)

        #if there is more than one ground truth
        if k == 'nat_lang_summ' or k == 'instructions':
            train_metrics_dict = one_to_many_compute_metrics(train_epoch_outputs[k], train_epoch_labels[k], all_episode_ids[k],
                train_dataset, k, eval_preds_to_save, fuller_metrics_to_save, global_epoch)

        else:

            train_metrics_dict = compute_metrics(train_epoch_outputs[k], train_epoch_labels[k],
                train_dataset, k, eval_preds_to_save, fuller_metrics_to_save, global_epoch)

        train_metrics_dict['loss'] = this_task_epoch_loss
        train_metrics_dict['acc'] = this_task_acc

        train_epoch_metrics_dicts[k] = train_metrics_dict

        writer.add_scalar('rougeL_f1/'+k+'/'+train_dataset.dataset_name, train_metrics_dict['full_rougeL'][1][2], global_epoch)
        writer.add_scalar('rougeL1_r/'+k+'/'+train_dataset.dataset_name, train_metrics_dict['full_rouge1'][1][1], global_epoch)
        writer.add_scalar('bleu/'+k+'/'+train_dataset.dataset_name, train_metrics_dict['bleu'], global_epoch)
        writer.add_scalar('prec1/'+k+'/'+train_dataset.dataset_name, train_metrics_dict['precisions'][0], global_epoch)

        if optimizer.param_groups[-1]['lr'] != None:
            writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], global_epoch)




    full_epoch_accuracy_dicts.append(train_epoch_accuracy_stats_dicts)
    all_record_dicts.append(train_epoch_accuracy_stats_dicts)
    all_metrics_dicts.append(train_epoch_metrics_dicts)

    unseen_epoch_stats = validate(model, valid_unseen_dataset, i, train_state, scheduler)

    logs_to_save.append(unseen_epoch_stats)

    #current_eval_dataset = "valid_seen_dataset"
    seen_epoch_stats = validate(model, valid_seen_dataset, i, train_state, scheduler)
    logs_to_save.append(seen_epoch_stats)



    val_acc_to_compare = seen_epoch_stats[args.main_task]['acc']
    unseen_val_acc_to_compare = unseen_epoch_stats[args.main_task]['acc']

    if args.use_scheduler:
        scheduler.step(val_acc_to_compare)

    print(global_epoch)


    print("main seen val acc: " + str(val_acc_to_compare))
    print("main unseen val acc: " + str(unseen_val_acc_to_compare))

    if val_acc_to_compare > best_val_acc:
        best_val_acc = val_acc_to_compare
        best_val_epoch = global_epoch
        unseen_val_acc_at_best_seen_val_epoch = unseen_val_acc_to_compare
        print("best seen val acc")
        if args.save_best_model:
            torch.save(model.state_dict(), output_directory+"best_seen_val_acc_model.pt")


    if unseen_val_acc_to_compare > best_unseen_val_acc:
        best_unseen_val_acc = unseen_val_acc_to_compare
        best_unseen_val_epoch = global_epoch
        print("best unseen val acc")
        if args.save_best_model:
            torch.save(model.state_dict(), output_directory+"best_unseen_val_acc_model.pt")

    print("epochs since best val acc: seen:" + str(global_epoch - best_val_epoch) + 
        " unseen: " + str(global_epoch - best_unseen_val_epoch))


    global_epoch += 1
    




def validate(model, dataset, outer_index, train_state, scheduler, limit_tasks_list = None, dataloader = None):
    global all_record_dicts
    global all_metrics_dicts

    if dataloader == None:
        dataloader = DataLoader(dataset=dataset, batch_size=args.valid_batch_size, shuffle = False, pin_memory = False,
            num_workers = args.num_workers, collate_fn=VisFeatsPadSequence())


    running_loss = 0.
    running_acc = 0.
    model.eval()



    running_losses = defaultdict(float)
    running_accs = defaultdict(float)

    valid_epoch_outputs = defaultdict(list)
    valid_epoch_labels = defaultdict(list)
    valid_epoch_questions = defaultdict(list)
    all_episode_ids = defaultdict(list)



    for i, batch_dicts in enumerate(dataloader):


        for task_name, task_dict in dataset.qa_task_descriptions.items():

            if limit_tasks_list:
                if task_name not in limit_tasks_list:
                    continue



            embgram_tensors = []
            input_images_tensors = []
            target_tensors = []
            text_input = []
            episode_ids = []

            for this_dict in batch_dicts:
                if task_name in this_dict:
                    if 'images' in this_dict:
                        input_images_tensors.append(this_dict['images'])
                    episode_ids.append(this_dict['episode_id'])
                    target_tensors.append(this_dict[task_name]['target_output'])
                    text_input.append(this_dict[task_name]['text_input'])


            if len(input_images_tensors) == 0:
                continue

            try:
                model_output = model(input_images_tensors, text_input, target_sequence = target_tensors, in_training_phase = False)

            except Exception as this_exception:
                print("validation exception")
                print(str(this_exception))
                print(episode_ids)
                IPython.embed()
                continue


            generated_output, full_model_output = model_output

            these_outputs = model.tokenizer.batch_decode(generated_output, skip_special_tokens=True)


            for w in these_outputs:
                valid_epoch_outputs[task_name].append(w)
            for w in target_tensors:
                valid_epoch_labels[task_name].append(w)
            for w in text_input:
                valid_epoch_questions[task_name].append(w)
            for ep_id in episode_ids:
                all_episode_ids[task_name].append(ep_id)


            loss = full_model_output['loss'].detach().item()

            running_losses[task_name] += loss

            del loss, model_output, full_model_output, generated_output





    num_minibatches = i + 1

    valid_epoch_accuracy_stats_dicts = {}
    valid_epoch_metrics_dicts = {}
    valid_full_accuracy_class_dicts = {}



    for k in valid_epoch_questions.keys():


        valid_epoch_dict = compute_accuracy_stats(valid_epoch_questions[k], valid_epoch_labels[k], 
            valid_epoch_outputs[k], all_episode_ids[k], k, dataset.dataset_name, global_epoch)

        valid_epoch_dict['epoch'] = global_epoch

        valid_accuracy_class = Step_by_step_compute_accuracy_stats(k, dataset.dataset_name)
        full_step_dict = valid_accuracy_class.compute_step(valid_epoch_questions[k], valid_epoch_labels[k],
            valid_epoch_outputs[k], all_episode_ids[k], global_epoch)
        valid_full_accuracy_class_dicts[k] = valid_accuracy_class.return_full_record_dict()


        valid_epoch_accuracy_stats_dicts[k] = valid_epoch_dict


        this_task_epoch_loss = running_losses[k]/num_minibatches




        writer.add_scalar('loss/'+k+'/'+dataset.dataset_name, this_task_epoch_loss, global_epoch)
        writer.add_scalar('acc/'+k+'/'+dataset.dataset_name, valid_epoch_dict['acc'], global_epoch)



        if k == 'nat_lang_summ' or k == 'instructions':
            valid_metrics_dict = one_to_many_compute_metrics(valid_epoch_outputs[k], valid_epoch_labels[k], all_episode_ids[k],
                dataset, k, eval_preds_to_save, fuller_metrics_to_save, global_epoch)

        else:

            valid_metrics_dict = compute_metrics(valid_epoch_outputs[k], valid_epoch_labels[k], 
                dataset, k, eval_preds_to_save, fuller_metrics_to_save, global_epoch)


        valid_metrics_dict['loss'] = this_task_epoch_loss
        valid_epoch_metrics_dicts[k] = valid_metrics_dict

        writer.add_scalar('rougeL_f1/'+k+'/'+dataset.dataset_name, valid_metrics_dict['full_rougeL'][1][2], global_epoch)
        writer.add_scalar('rougeL1_r/'+k+'/'+dataset.dataset_name, valid_metrics_dict['full_rouge1'][1][1], global_epoch)
        writer.add_scalar('bleu/'+k+'/'+dataset.dataset_name, valid_metrics_dict['bleu'], global_epoch)
        writer.add_scalar('prec1/'+k+'/'+dataset.dataset_name, valid_metrics_dict['precisions'][0], global_epoch)

    all_record_dicts.append(valid_epoch_accuracy_stats_dicts)
    all_metrics_dicts.append(valid_epoch_metrics_dicts)
    full_epoch_accuracy_dicts.append(valid_full_accuracy_class_dicts)

    return valid_epoch_accuracy_stats_dicts


def read_alfred_dict(dataset_dict_files, skip_certain_ids = None):
    
    dict_to_return = {}


    first_dataset_to_load = True

    for dataset_file in dataset_dict_files:


        f=open(dataset_file, 'rb')
        dataset_dict = pickle.load(f)
        f.close()


        if first_dataset_to_load:

            for k,v in dataset_dict.items():

                if skip_certain_ids:
                    if k in skip_certain_ids:
                        continue

                dict_to_return[k] = v


        else:
            
            for k,v in dataset_dict.items():


                if skip_certain_ids:
                    if k in skip_certain_ids:
                        continue

                dict_to_return[k].update(v)

        first_dataset_to_load = False

    return dict_to_return




set_seed_everywhere(args.seed)

if args.static_validate:


    valid_seen_dataset = DictMultitaskDataset(valid_seen_collected_dicts,
        'valid_seen', qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions, valid_seen_action_counter, 
        allowed_ids = allowed_subset_of_valid_seen_ids,
        static_validate_dict = static_validate_dicts['valid_seen'], num_static_examples = 10)

    valid_unseen_dataset = DictMultitaskDataset(valid_unseen_collected_dicts,
        'valid_unseen', qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions,
        valid_unseen_action_counter, allowed_ids = allowed_subset_of_valid_unseen_ids,
        static_validate_dict = static_validate_dicts['valid_unseen'], num_static_examples = 10)

    for dataset in [valid_unseen_dataset, valid_seen_dataset]:

        for k,v in dataset.id_dataset_dicts.items():
            v['clean_low_pddl'].append('stop')

else:


    valid_unseen_collected_dicts = read_alfred_dict(valid_unseen_dicts, skip_certain_ids=valid_unseen_ids_to_skip)
    valid_seen_collected_dicts = read_alfred_dict(valid_seen_dicts)
    train_collected_dicts = read_alfred_dict(train_dicts, skip_certain_ids = train_ids_to_skip)


    train_dataset = DictMultitaskDataset(train_collected_dicts,
        'train', qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions, train_action_counter, 
        allowed_ids = allowed_subset_of_train_ids)

    valid_seen_dataset = DictMultitaskDataset(valid_seen_collected_dicts,
        'valid_seen', qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions, 
        valid_seen_action_counter, allowed_ids = allowed_subset_of_valid_seen_ids)

    valid_unseen_dataset = DictMultitaskDataset(valid_unseen_collected_dicts,
        'valid_unseen', qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions, 
        valid_unseen_action_counter, allowed_ids = allowed_subset_of_valid_unseen_ids)



    for dataset in [train_dataset, valid_unseen_dataset, valid_seen_dataset]:

        for k,v in dataset.id_dataset_dicts.items():
            v['clean_low_pddl'].append('stop')

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = False,
        num_workers = args.num_workers, collate_fn=VisFeatsPadSequence())





metric = load_metric("rouge")
bleu = load_metric("bleu")


train_state = make_train_state(args)


model = SummModel(transformer_model = transformer_model)



if args.load_pretrained_state_dict or args.static_validate:
    print("loading pretrained model:")
    print(saved_model_state_dict)
    model.load_state_dict(torch.load(saved_model_state_dict, map_location='cpu'), strict=False)


if args.load_conv_layers:

    convs_state_dict = torch.load(args.load_conv_layers_model, map_location='cpu')

    with torch.no_grad():
            model.cnn_encoder.conv3.bias.copy_(convs_state_dict['cnn_encoder.conv3.bias'])
            model.cnn_encoder.conv2.bias.copy_(convs_state_dict['cnn_encoder.conv2.bias'])
            model.cnn_encoder.conv1.bias.copy_(convs_state_dict['cnn_encoder.conv1.bias'])

            model.cnn_encoder.conv3.weight.copy_(convs_state_dict['cnn_encoder.conv3.weight'])
            model.cnn_encoder.conv2.weight.copy_(convs_state_dict['cnn_encoder.conv2.weight'])
            model.cnn_encoder.conv1.weight.copy_(convs_state_dict['cnn_encoder.conv1.weight'])

            model.cnn_encoder.bn1.weight.copy_(convs_state_dict['cnn_encoder.bn1.weight'])
            model.cnn_encoder.bn2.weight.copy_(convs_state_dict['cnn_encoder.bn2.weight'])
            model.cnn_encoder.bn3.weight.copy_(convs_state_dict['cnn_encoder.bn3.weight'])

            model.cnn_encoder.bn1.bias.copy_(convs_state_dict['cnn_encoder.bn1.bias'])
            model.cnn_encoder.bn2.bias.copy_(convs_state_dict['cnn_encoder.bn2.bias'])
            model.cnn_encoder.bn3.bias.copy_(convs_state_dict['cnn_encoder.bn3.bias'])


model.transformer_model.config.dropout_rate = 0.1


if args.optimizer_type == 'adam':

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                       mode='max', factor=0.75,
                                       patience=20, min_lr = .00002, cooldown = 0)
    args.use_scheduler = True


elif args.optimizer_type == 'adafactor':

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)


    scheduler = None
    args.use_scheduler = False



if args.static_validate:
    unseen_epoch_stats = alidate(model, valid_unseen_dataset, 0, train_state, scheduler)

    logs_to_save.append(unseen_epoch_stats)

    #current_eval_dataset = "valid_seen_dataset"
    seen_epoch_stats = validate(model, valid_seen_dataset, 0, train_state, scheduler)
    logs_to_save.append(seen_epoch_stats)


    val_acc_to_compare = seen_epoch_stats[args.main_task]['acc']
    unseen_val_acc_to_compare = unseen_epoch_stats[args.main_task]['acc']

    print("seen val acc: " + str(val_acc_to_compare))
    print("unseen val acc: " + str(unseen_val_acc_to_compare))

    save_records()
    IPython.embed()



for i in range(args.number_of_epochs):

    train(model, train_dataset, valid_unseen_dataset, valid_seen_dataset,
        train_state, train_dataloader, optimizer, scheduler)

    writer.flush()
    save_records()
    save_general_checkpoint()


IPython.embed()


