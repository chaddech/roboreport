import random
import numpy as np
import torch
from torch.utils.data import Dataset
import IPython
from roboreport_helper_files.roboreport_helper_methods import *
from roboreport_helper_files.generate_questions import get_positive_and_negative_questions
from roboreport_helper_files.generate_questions import get_negative_action_questions, get_positive_action_questions
from roboreport_helper_files.generate_questions import make_nat_lang_right_before_questions, make_nat_lang_right_after_questions
from roboreport_helper_files.roboreport_config_file import args


class DictMultitaskDataset(Dataset):
    def __init__(self, id_dataset_dicts,
        dataset_name, qa_task_descriptions, main_qa_task_descriptions, secondary_task_descriptions, 
        action_counter_dict, allowed_ids = None,
        task_type = None, allowed_objects=None, static_validate_dict = None, num_static_examples = None):


        self.action_counter_dict = action_counter_dict

        self.task_type = task_type

        if args.use_clip_feats:
            self.vis_vector_len = 7 * 7 * 2048
        elif args.use_trad_clip_feats:
            self.vis_vector_len = 1024

        self.focus_on_one_scene = False
        self.loaded_source_tensor = None

        self.main_qa_task_descriptions = main_qa_task_descriptions
        self.secondary_task_descriptions = secondary_task_descriptions

        self.focus_on_questions = None
        self.use_sampling_for_negatives = args.use_sampling_for_negatives

        self.return_yes_question = False
        self.return_no_question = False

        self.episode_id_to_data_index = {}

        self.dataset_name = dataset_name

        self.allowed_objects = allowed_objects

        if allowed_ids:

            self.allowed_ids = []

            for allowed_id in allowed_ids:
                if allowed_id in id_dataset_dicts:
                    self.allowed_ids.append(allowed_id)

        else: 
            self.allowed_ids = list(id_dataset_dicts.keys())


        if args.limit_dataset_size:
            self.len = args.limit_dataset_size
        else:
            self.len = len(self.allowed_ids)


        self.id_dataset_dicts = id_dataset_dicts
        self.qa_task_descriptions = qa_task_descriptions
        self.task_keys = list(qa_task_descriptions.keys())


        if static_validate_dict:
            self.static_validate_dict = static_validate_dict
            self.static_valid_ids_and_indices = []
            for k,v in static_validate_dict.items():
                for act_index in v:
                    self.static_valid_ids_and_indices.append((k,act_index))
            self.len = len(self.static_valid_ids_and_indices)
        else:
            self.static_validate_dict = False

        print(dataset_name)
        print(self.len)


    def set_new_allowed_ids(self, new_allowed_ids):

            self.allowed_ids = []

            for allowed_id in new_allowed_ids:
                if allowed_id in id_dataset_dicts:
                    self.allowed_ids.append(allowed_id)

            if LIMIT_DATASET_SIZE:
                self.len = LIMIT_DATASET_SIZE
            else:
                self.len = len(self.allowed_ids)


    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(self.EOS_token)
        #removed cuda below
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def sourceTensor(self, sentence):

        labels = self.tokenizer(sentence).input_ids
        labels = torch.tensor(labels)
        return labels

    def tensorFromVisFeats(self, vector):

        vector = vector.float()

        source_tensor = vector.reshape((vector.shape[0], self.vis_vector_len))
        #source_tensor_w_SOS = torch.cat((self.vector_SOS_token, source_tensor), 0)
        #source_tensor_w_SOS_and_EOS = torch.cat((source_tensor_w_SOS, self.vector_EOS_token), 0)
        return source_tensor



    def get_simple_object_yes_no_question(self, questions_dict, num_to_return=1):

        if self.return_yes_question:
            return_yes = True
        elif self.return_no_question:
            return_yes = False
        else:
            if random.getrandbits(1) == 1:
                return_yes = True
            else:
                return_yes = False

        if return_yes:
            question = random.choices(questions_dict['objects_seen_set_list'], k=num_to_return)
            answer = "yes"
        else:
            if self.use_sampling_for_negatives:
                question = random.choices(list(questions_dict['objects_not_seen'].keys()),
                    weights=list(questions_dict['objects_not_seen'].values()), k=num_to_return)

            else:
                question = random.choices(list(questions_dict['objects_not_seen'].keys()), k=num_to_return)
            answer = "no"

        if num_to_return == 1:
            question = question[0]

        question = "was there a " + question 


        return question, answer




    def get_action_either_or_question(self, episode_dict, num_to_return=1):

        yes_questions, no_questions = get_positive_and_negative_questions(self.action_counter_dict, 
            episode_dict['clean_high_pddl'])


        yes_question = np.random.choice(yes_questions)
        no_question = np.random.choice(no_questions)

        if no_question != None:


            if bool(random.getrandbits(1)):

                question = 'did you ' + yes_question + ' or ' + no_question + '?'
            else:
                question = 'did you ' + no_question + ' or ' + yes_question + '?'


            answer = yes_question

        else:
            question = None
            answer = None 


        return question, answer



    def get_object_either_or_question(self, questions_dict, num_to_return=1):

        if random.getrandbits(1) == 1:
            return_positive_first = True
        else:
            return_positive_first = False

        positive_object = random.choices(questions_dict['objects_seen_set_list'], k=num_to_return)
        positive_object = positive_object[0]

        if self.use_sampling_for_negatives:
            negative_object = random.choices(list(questions_dict['objects_not_seen'].keys()),
                weights=list(questions_dict['objects_not_seen'].values()), k=num_to_return)

        else:
            negative_object = random.choices(list(questions_dict['objects_not_seen'].keys()), k=num_to_return)

        negative_object = negative_object[0]


        if return_positive_first:
            return positive_object, negative_object, positive_object
        else:
            return negative_object, positive_object, positive_object
        

    def get_simple_action_yes_no_questions(self, questions_dict):

        if self.return_yes_question:
            return_yes = True
        elif self.return_no_question:
            return_yes = False
        else:
            if random.getrandbits(1) == 1:
                return_yes = True
            else:
                return_yes = False

        if return_yes:
            question = random.choice(questions_dict['objects_seen_set_list'])
            question = "was there a " + question 
            answer = "yes"
        else:
            question = random.choice(list(questions_dict['objects_not_seen'].keys()))
            question = "was there a " + question 
            answer = "no"

        return question, answer



    def get_negative_nat_lang_action(self, pddl):
        need_sample = True
        while need_sample:
            indx = np.random.choice(range(self.len))
            this_episode_dict = self.id_dataset_dicts[self.allowed_ids[indx]]
            outer_indx = np.random.choice(range(len(this_episode_dict['high_desc_annotation_individual_sentences'])))
            line_indx = np.random.choice(range(len(this_episode_dict['high_desc_annotation_individual_sentences'][outer_indx])))
            if this_episode_dict['clean_high_pddl'][line_indx] not in pddl:
                question_line = this_episode_dict['high_desc_annotation_individual_sentences'][outer_indx][line_indx]
                need_sample = False
        return question_line




    def __len__(self):
        return self.len

    def __getitem__(self, idx):


        if self.static_validate_dict:
            episode_dict = self.id_dataset_dicts[self.static_valid_ids_and_indices[idx][0]]
            action_index_to_test = self.static_valid_ids_and_indices[idx][1]
        else:
            episode_dict = self.id_dataset_dicts[self.allowed_ids[idx]]

        batch_dict_to_return = {}

        batch_dict_to_return['dataset_name'] = self.dataset_name



        for k,v in self.qa_task_descriptions.items():

            this_dict_to_return = {}

            if v['name'] == 'nat_lang_action_yes_no':

                if bool(random.getrandbits(1)):
                    outer_indx = np.random.choice(range(len(episode_dict['high_desc_annotation_individual_sentences'])))
                    line_indx = np.random.choice(range(len(episode_dict['high_desc_annotation_individual_sentences'][outer_indx])))
                    instr_line = episode_dict['high_desc_annotation_individual_sentences'][outer_indx][line_indx][:-1]
                    line_to_return = " ".join(instr_line)

                    i1 = episode_dict['high_desc_annotation_individual_sentences'][indx]

                    this_dict_to_return['target_output'] = 'yes'
                    this_dict_to_return['text_input'] = 'did you ' + line_to_return + '?'
                else:

                    p1 =episode_dict['clean_high_pddl']

                    no_question = self.get_negative_nat_lang_action(p1)
                    no_question = " ".join(no_question[:-1])

                    this_dict_to_return['text_input'] = 'did you ' + no_question + '?'

                    this_dict_to_return['target_output'] = 'no'


            elif v['name'] == 'instructions':

                this_dict_to_return['text_input'] = v['summarize_instruction']
                output_text = random.choice(episode_dict[v['target_output']])

                output_list = []
                for sentence in output_text:
                    for word in sentence:
                        output_list.append(word)


                output_string = " ".join(output_list)


                this_dict_to_return['target_output'] = output_string


            elif v['name'] == 'pddl_summ':
                this_dict_to_return['text_input'] = v['summarize_instruction']

                output_list = episode_dict[v['target_output']]

                if args.translate_pddl_to_english:
                    output_string = " , ".join(output_list)
                else:                
                    output_string = " ".join(output_list)

                this_dict_to_return['target_output'] = output_string

            elif v['name'] == 'simple_object_yes_no':
            
                question, answer = self.get_simple_object_yes_no_question(episode_dict)

                if question[0] in vowels:
                    this_dict_to_return['text_input'] = 'was there an ' + question + ' ?'

                else:
                    this_dict_to_return['text_input'] = 'was there a ' + question + ' ?'

                this_dict_to_return['target_output'] = answer


            elif v['name'] == 'object_either_or':
            
                object_one, object_two, answer = self.get_object_either_or_question(episode_dict)

                if object_one[0] in vowels:

                    question_text = 'was there an ' + object_one 

                else:

                    question_text = 'was there a ' + object_one

                if object_two[0] in vowels:

                    question_text += ' or an ' + object_two + ' ?'

                else:
                    question_text +=  ' or a ' + object_two + ' ?'


                this_dict_to_return['text_input'] = question_text
                this_dict_to_return['target_output'] = answer



            elif v['name'] == 'nat_lang_summ':

                this_dict_to_return['text_input'] = v['nat_lang_summarize_instruction']

                output_list = random.choice(episode_dict[v['target_output']])
                output_string = " ".join(output_list)

                this_dict_to_return['target_output'] = output_string



            elif v['name'] == 'low_actions':

                this_dict_to_return['text_input'] = v['full_narrate_instruction']

                output_list = episode_dict[v['target_output']]
                output_string = " ".join(output_list)

                this_dict_to_return['target_output'] = output_string



            elif v['name'] == 'simple_action_yes_no':
                yes_questions, no_questions = get_positive_and_negative_questions(self.action_counter_dict, 
                    episode_dict['clean_high_pddl'])
                if bool(random.getrandbits(1)):
                    yes_question = np.random.choice(yes_questions)
                    this_dict_to_return['target_output'] = 'yes'
                    this_dict_to_return['text_input'] = 'did you ' + yes_question + ' ?'
                else:
                    no_question = np.random.choice(no_questions)
                    this_dict_to_return['target_output'] = 'no'
                    if no_question != None:
                        this_dict_to_return['text_input'] = 'did you ' + no_question + ' ?'



            elif v['name'] == 'action_either_or':
                question, answer = self.get_action_either_or_question(episode_dict)

                if question != None:

                    this_dict_to_return['text_input'] = question
                    this_dict_to_return['target_output'] = answer


            elif v['name'] == 'before_actions_yes_no':
                yes_q = False
                no_q = False

                if bool(random.getrandbits(1)):

                    yes_q = True
                    if len(episode_dict['yes_before_questions']) == 0:
                        yes_q = False
                        if len(episode_dict['not_before_questions']) != 0:
                            no_q = True
                else:
                    no_q = True
                    if len(episode_dict['not_before_questions']) == 0:
                        no_q = False
                        if len(episode_dict['yes_before_questions']) != 0:
                            yes_q = True


                if yes_q:
                    yes_question_idx = np.random.choice(len(episode_dict['yes_before_questions']))
                    yes_question = episode_dict['yes_before_questions'][yes_question_idx]
                    this_dict_to_return['target_output'] = 'yes'
                    this_dict_to_return['text_input'] = 'Did you ' + yes_question[0] +' before ' +yes_question[1] + ' ?'

                if no_q:
                    no_question_idx = np.random.choice(len(episode_dict['not_before_questions']))
                    no_question = episode_dict['not_before_questions'][no_question_idx]
                    this_dict_to_return['target_output'] = 'no'
                    this_dict_to_return['text_input'] = 'Did you ' + no_question[0] +' before ' +no_question[1] + ' ?'

            elif v['name'] == 'before_actions_output_name':
                first_action_before = False
                second_action_before = False

                if bool(random.getrandbits(1)):

                    first_action_before = True
                    if len(episode_dict['yes_before_questions']) == 0:
                        first_action_before = False
                        if len(episode_dict['not_before_questions']) != 0:
                            second_action_before = True
                else:
                    second_action_before = True
                    if len(episode_dict['not_before_questions']) == 0:
                        second_action_before = False
                        if len(episode_dict['yes_before_questions']) != 0:
                            first_action_before = True




                if first_action_before:
                    first_action_before_idx = np.random.choice(len(episode_dict['yes_before_questions']))
                    first_action_before_question = episode_dict['yes_before_questions'][first_action_before_idx]
                    this_dict_to_return['target_output'] = first_action_before_question[0]
                    this_dict_to_return['text_input'] = 'Which did you do first, ' + first_action_before_question[0] +' or ' +first_action_before_question[1] + ' ?'

                if second_action_before:
                    second_action_before_idx = np.random.choice(len(episode_dict['not_before_questions']))
                    second_action_before_question = episode_dict['not_before_questions'][second_action_before_idx]
                    this_dict_to_return['target_output'] = second_action_before_question[1]
                    this_dict_to_return['text_input'] = 'Which did you do first, ' + second_action_before_question[0] +' or ' +second_action_before_question[1] + ' ?'


            elif v['name'] == 'right_before_questions':

                if len(episode_dict['right_before_questions']) != 0:

                    question_idx = np.random.choice(len(episode_dict['right_before_questions']))
                    question = episode_dict['right_before_questions'][question_idx]
                    this_dict_to_return['target_output'] = question[1]
                    this_dict_to_return['text_input'] = 'What did you do just before ' + question[0] +' ?'


            elif v['name'] == 'right_after_questions':

                if len(episode_dict['right_after_questions']) != 0:

                    question_idx = np.random.choice(len(episode_dict['right_after_questions']))
                    question = episode_dict['right_after_questions'][question_idx]
                    this_dict_to_return['target_output'] = question[1]
                    this_dict_to_return['text_input'] = 'What did you do just after ' + question[0] +' ?'



            elif v['name'] == 'nat_lang_right_before_questions':

                p1 =episode_dict['clean_high_pddl']
                indx = np.random.choice(range(len(episode_dict['high_desc_annotation_individual_sentences'])))
                i1 = episode_dict['high_desc_annotation_individual_sentences'][indx]
                right_before_qs = make_nat_lang_right_before_questions(p1, i1)
                if len(right_before_qs) != 0:

                    question_idx = np.random.choice(range(len(right_before_qs)))
                    question = right_before_qs[question_idx]
                    this_dict_to_return['target_output'] = question[1]
                    q_sentence = " ".join(question[0])
                    this_dict_to_return['text_input'] = 'What did you do just before ' + q_sentence +' ?'


            elif v['name'] == 'nat_lang_right_after_questions':


                p1 =episode_dict['clean_high_pddl']
                indx = np.random.choice(range(len(episode_dict['high_desc_annotation_individual_sentences'])))
                i1 = episode_dict['high_desc_annotation_individual_sentences'][indx]
                right_after_qs = make_nat_lang_right_after_questions(p1, i1)
                if len(right_after_qs) != 0:

                    question_idx = np.random.choice(range(len(right_after_qs)))
                    question = right_after_qs[question_idx]
                    this_dict_to_return['target_output'] = question[1]
                    q_sentence = " ".join(question[0])

                    this_dict_to_return['text_input'] = 'What did you do just after ' + q_sentence +' ?'


            if len(this_dict_to_return) > 0:

                batch_dict_to_return[v['name']] = this_dict_to_return 

                if args.translate_pddl_to_english:


                    new_input_text = translate_pddl_to_english(this_dict_to_return['text_input'].split())
                    new_output_text = translate_pddl_to_english(this_dict_to_return['target_output'].split())


                    this_dict_to_return['text_input'] = new_input_text

                    this_dict_to_return['target_output'] = new_output_text


        # load episode clip image feats
        if args.use_clip_feats:

            raw_file_name = episode_dict['resnet_features_file_name'].split("/")
            raw_file_name = raw_file_name[5:]


            raw_file_name[-1] = alfred_clip_file_name
            prefix_for_clip = ["",  "mnt", "oslo", "alfred_stuff", "alfred_clip_feats"]
            source_file = prefix_for_clip + raw_file_name
            source_file = "/".join(source_file)

        elif args.use_trad_clip_feats:

            raw_file_name = episode_dict['resnet_features_file_name'].split("/")
            raw_file_name = raw_file_name[5:]
            prefix_for_clip = ["",  "mnt", "oslo", "alfred_stuff", "alfred_clip_feats"]
            source_file = prefix_for_clip + raw_file_name
            raw_file_name[-1] = alfred_clip_file_name
            source_file = prefix_for_clip + raw_file_name
            source_file = "/".join(source_file)

        else: 
            #Define location for Alfred feats
            prefix_for_alfred_feats = ""
            original_feat_file_name = episode_dict['resnet_features_file_name'].split("/")[5:]
            source_file = prefix_for_alfred_feats+original_feat_file_name
            source_file = "/".join(source_file)

        if self.focus_on_one_scene:
            if self.loaded_source_tensor:
                source_tensor = self.loaded_source_tensor
            else: 
                source_tensor = torch.load(source_file)
                source_tensor = self.tensorFromVisFeats(source_tensor)
                self.loaded_source_tensor = source_tensor

        else:
            source_tensor = torch.load(source_file, map_location='cpu')
            if not args.use_trad_clip_feats:
                source_tensor = self.tensorFromVisFeats(source_tensor)

        
        batch_dict_to_return['images'] = source_tensor


        batch_dict_to_return['episode_id'] = episode_dict['task_id']


        return batch_dict_to_return
