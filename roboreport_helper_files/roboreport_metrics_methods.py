from collections import Counter, defaultdict
import pickle
from datasets import load_metric
import nltk
import string

metric = load_metric("rouge")
bleu = load_metric("bleu")




action_words = set(('gotolocation', 'pickupobject', 'putobject', 'coolobject', 'heatobject', 'cleanobject', 
'toggleobject', 'sliceobject', 'go', 'pick', 'put', 'cool', 'heat', 'clean', 'toggle', 'slice', 'moveahead', 
'rotateleft', 'rotateright', 'lookdown', 'lookup', 'closeobject', 'openobject'))



DICT_OF_VOCAB_IN_EACH_SET_FILE = 'lower_case_dict_of_vocab_in_each_set.pkl'
f = open(DICT_OF_VOCAB_IN_EACH_SET_FILE, 'rb')
dict_of_vocab_in_each_set = pickle.load(f)
f.close() 


prepositions = set(('in', 'on', 'up', 'to', 'the'))

tasks_outputing_non_yes_no_answers = ['pddl_summ', 'right_after_questions', 'right_before_questions',
                'object_either_or', 'action_either_or', 'before_actions_output_name', 'low_actions', 'instructions',
                'nat_lang_right_before_questions', 'nat_lang_right_after_questions', 'act_from_goal', 'act_from_history_only',
                'act_from_embgram_and_history', 'act_from_instructions_w_history', 'act_from_goal_w_history',
                'act_from_goal', 'act_from_embgram', 'act_from_history_only']


def compute_metrics(predictions, labels, dataset, task_type, eval_preds_to_save, fuller_metrics_to_save, global_epoch):

    this_dataset = dataset
    current_eval_dataset = dataset.dataset_name

    eval_preds_to_save.append((task_type, dataset.dataset_name, predictions, labels))

    result = metric.compute(predictions=predictions, references=labels, use_stemmer=True)
    
    word_decoded_preds = [nltk.word_tokenize(pred.strip()) for pred in predictions]
    word_decoded_labels = [[nltk.word_tokenize(label.strip())] for label in labels]


    bleu_result = None
    try:
        bleu_result = bleu.compute(predictions=word_decoded_preds, references=word_decoded_labels)

    except:
        print("exception at bleu result")
        IPython.embed()


    fuller_result_to_keep = {}
    for k in result.keys():
        fuller_result_to_keep["full_"+k] = result[k]

    result_to_return = {key: value.mid.fmeasure * 100 for key, value in result.items()}



    
    result_to_return["dataset"] = dataset.dataset_name
    result_to_return['global_epoch'] = global_epoch

    if bleu_result!= None:
        result_to_return.update(bleu_result)

    fuller_result_to_keep.update(result_to_return)
    fuller_result_to_keep['task_type'] = task_type
    fuller_result_to_keep['dataset_len'] = this_dataset.__len__()
    fuller_metrics_to_save.append(fuller_result_to_keep)



    return fuller_result_to_keep



def one_to_many_compute_metrics(predictions, labels, episode_ids, dataset, task_name,
    eval_preds_to_save, fuller_metrics_to_save, global_epoch):

    eval_preds_to_save.append((task_name, dataset.dataset_name, episode_ids, predictions, labels))

    
    decoded_preds = predictions
    decoded_labels = labels

    word_decoded_preds = [nltk.word_tokenize(pred.strip()) for pred in decoded_preds]
    word_decoded_labels = [[nltk.word_tokenize(label.strip())] for label in decoded_labels]

    new_word_preds = []
    new_word_labels = []


    this_dataset = dataset



    if task_name == 'nat_lang_summ':
        for x in episode_ids:
            new_word_labels.append(dataset.id_dataset_dicts[x]['task_desc_summaries'])
    elif task_name == 'instructions':
        for x in episode_ids:
            all_annotations =  dataset.id_dataset_dicts[x]['high_desc_annotation_individual_sentences']
            this_episode_all_labels = []
            for raw_annotator in all_annotations:
                this_annotators_text = []
                for sentence in raw_annotator:
                    for word in sentence:
                        this_annotators_text.append(word)
                this_episode_all_labels.append(this_annotators_text)

            new_word_labels.append(this_episode_all_labels)

    bleu_result = bleu.compute(predictions=word_decoded_preds, references=new_word_labels)


    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    fuller_result_to_keep = {}
    for k in result.keys():
        fuller_result_to_keep["full_"+k] = result[k]

    result_to_return = {key: value.mid.fmeasure * 100 for key, value in result.items()}



    result_to_return["dataset"] = dataset.dataset_name
    result_to_return['global_epoch'] = global_epoch

    result_to_return.update(bleu_result)

    fuller_result_to_keep.update(result_to_return)
    fuller_result_to_keep['dataset_len'] = dataset.__len__()
    fuller_result_to_keep['task_type'] = task_name
    fuller_metrics_to_save.append(fuller_result_to_keep)


    return fuller_result_to_keep




def compute_accuracy_stats(epoch_questions, epoch_labels, epoch_outputs,
    episode_ids, task_name, dataset_name, global_epoch):

    this_record_dict = {}

    number_of_training_examples_correct = 0


    accurate_questions = []
    inaccurate_questions = []
    accurate_output_answers = []
    inaccurate_output_answers = []
    correct_answers_for_inaccurate_output_answers = []
    correct_episode_ids = []
    incorrect_episode_ids = []

    extra_words_in_epoch = {}
    missing_words_in_epoch = {}



    extra_words_in_epoch_counter = Counter()
    missing_words_in_epoch_counter = Counter()



    number_of_examples_correct = 0

    for i in range(len(epoch_labels)):
        if epoch_labels[i] == epoch_outputs[i]:
            number_of_examples_correct += 1
            accurate_questions.append(epoch_questions[i])
            accurate_output_answers.append(epoch_outputs[i])
            correct_episode_ids.append(episode_ids[i])

        else:
            inaccurate_output_answers.append(epoch_outputs[i])
            correct_answers_for_inaccurate_output_answers.append(epoch_labels[i])
            inaccurate_questions.append(epoch_questions[i])
            incorrect_episode_ids.append(episode_ids[i])

            if task_name == tasks_outputing_non_yes_no_answers:
                these_outputs = set(epoch_outputs[i].split())

                these_labels = set(epoch_labels[i].split())
                these_missing_words = these_labels - these_outputs
                these_extra_words = these_outputs - these_labels
                extra_words_in_epoch[episode_ids[i]] = these_extra_words
                missing_words_in_epoch[episode_ids[i]] = these_missing_words
                for w in these_missing_words:
                    missing_words_in_epoch_counter[w] += 1
                for w in these_extra_words:
                    if w in dict_of_vocab_in_each_set['train']:
                        extra_words_in_epoch_counter[w] += 1





    
    epoch_acc = number_of_examples_correct / len(epoch_labels)



    if task_name in tasks_outputing_non_yes_no_answers:
        this_record_dict['missing_words_in_epoch'] = missing_words_in_epoch
        this_record_dict['extra_words_in_epoch'] = extra_words_in_epoch
        this_record_dict['missing_words_in_epoch_counter'] = missing_words_in_epoch_counter
        this_record_dict['extra_words_in_epoch_counter'] = extra_words_in_epoch_counter




    this_record_dict['dataset_name'] = dataset_name
    this_record_dict['task_name'] = task_name
    this_record_dict['epoch'] = global_epoch
    this_record_dict['acc'] = epoch_acc
    this_record_dict['accurate_questions'] = accurate_questions
    this_record_dict['accurate_output_answers'] = accurate_output_answers
    this_record_dict['inaccurate_questions'] = inaccurate_questions 
    this_record_dict['inaccurate_output_answers'] = inaccurate_output_answers
    this_record_dict['correct_episode_ids'] = correct_episode_ids
    this_record_dict['incorrect_episode_ids'] = incorrect_episode_ids
    this_record_dict['correct_answers_for_inaccurate_output_answers'] = correct_answers_for_inaccurate_output_answers

    return this_record_dict



class Step_by_step_compute_accuracy_stats:

    def __init__(self, task_name, dataset_name):

        self.record_dict = {}


        self.record_dict['accurate_questions'] = []

        self.record_dict['inaccurate_questions'] = []
        self.record_dict['accurate_output_answers'] = []
        self.record_dict['inaccurate_output_answers'] = []

        self.record_dict['correct_episode_ids'] = []
        self.record_dict['incorrect_episode_ids'] = []

        self.record_dict['number_of_examples_correct'] = 0
        self.record_dict['total_number_of_examples'] = 0

        self.number_of_examples_correct = 0

        self.record_dict['correct_answers_for_inaccurate_output_answers'] = []


        self.task_name = task_name
        self.dataset_name = dataset_name

        self.record_dict['dataset_name'] = dataset_name
        self.record_dict['task_name'] = task_name

        self.record_dict['missing_words_in_epoch_counter'] = Counter()
        self.record_dict['extra_words_in_epoch_counter'] = Counter()
        self.record_dict['missing_objects_in_epoch_counter'] = Counter()
        self.record_dict['missing_actions_in_epoch_counter'] = Counter()
        self.record_dict['extra_objects_in_epoch_counter'] = Counter()
        self.record_dict['extra_actions_in_epoch_counter'] = Counter()


        self.record_dict['accurate_words_in_epoch_counter'] = Counter()
        self.record_dict['label_words_in_epoch_counter'] = Counter()

        self.record_dict['missing_actions_in_episodes'] = {}
        self.record_dict['missing_objects_in_episodes'] = {}
        self.record_dict['extra_actions_in_episodes'] = {}
        self.record_dict['extra_objects_in_episodes'] = {}

        self.record_dict['raw_output_by_id'] = {}

        self.record_dict['edit_distances_in_epoch'] = []
        self.record_dict['edit_distance_in_episodes'] = {}



    def compute_step(self, epoch_questions, epoch_labels, epoch_outputs, episode_ids, global_epoch):



        this_record_dict = {}

        number_of_training_examples_correct = 0


        accurate_questions = []
        inaccurate_questions = []
        accurate_output_answers = []
        inaccurate_output_answers = []
        correct_answers_for_inaccurate_output_answers = []
        correct_episode_ids = []
        incorrect_episode_ids = []


        extra_words_in_episodes_counter = Counter()
        missing_words_in_episodes_counter = Counter()

        accurate_words_in_epoch_counter = Counter()
        label_words_in_epoch_counter = Counter()
        missing_words_in_epoch_counter = Counter()
        extra_words_in_epoch_counter = Counter()

        extra_actions_in_epoch_counter = Counter()
        extra_objects_in_epoch_counter = Counter()
        missing_actions_in_epoch_counter = Counter()
        missing_objects_in_epoch_counter = Counter()


        accurate_words_in_episodes_counter = Counter()


        extra_actions_in_episodes = defaultdict(Counter)
        missing_actions_in_episodes = defaultdict(Counter)
        missing_objects_in_episodes = defaultdict(Counter)
        extra_objects_in_episodes = defaultdict(Counter)

        missing_actions_in_episodes_counter = defaultdict(Counter)
        missing_objects_in_episodes_counter = defaultdict(Counter)

        extra_actions_in_episodes_counter = defaultdict(Counter)
        extra_objects_in_episodes_counter = defaultdict(Counter)


        in_vocab_question_counts_in_epoch = Counter()

        edit_distance_in_episodes = {}

        in_vocab_output_counts_in_epoch = Counter()


        raw_output_by_id = {}


        number_of_examples_correct = 0

        for i in range(len(epoch_labels)):
            raw_output_by_id[episode_ids[i]] = epoch_outputs[i]



            if epoch_labels[i] == epoch_outputs[i]:

                number_of_examples_correct += 1
                self.number_of_examples_correct += 1

                accurate_questions.append(epoch_questions[i])
                self.record_dict['accurate_questions'].append(epoch_questions[i])


                accurate_output_answers.append(epoch_outputs[i])
                self.record_dict['accurate_output_answers'].append(epoch_outputs[i])

                self.record_dict['correct_episode_ids'].append(episode_ids[i])
                correct_episode_ids.append(episode_ids[i])




                if self.task_name in tasks_outputing_non_yes_no_answers:
                    these_in_order_outputs = epoch_outputs[i].split()
                    these_in_order_labels = epoch_labels[i].split()

                    for w in these_in_order_outputs:
                        self.record_dict['accurate_words_in_epoch_counter'][w] += 1
                        self.record_dict['label_words_in_epoch_counter'][w] += 1

                        label_words_in_epoch_counter[w] += 1
                        in_vocab_output_counts_in_epoch[w] += 1


                elif self.task_name in ['simple_object_yes_no', 'simple_action_yes_no', 'before_actions_yes_no', 
                'nat_lang_action_yes_no']:
                    these_in_order_questions = epoch_questions[i]
                   
                    these_in_order_questions = these_in_order_questions.translate(str.maketrans('', '', string.punctuation))

                    these_in_order_questions = these_in_order_questions.split()

                    for w in these_in_order_questions:
                        if w in dict_of_vocab_in_each_set['train']:
                            self.record_dict['accurate_words_in_epoch_counter'][w] += 1
                            self.record_dict['label_words_in_epoch_counter'][w] += 1

                            label_words_in_epoch_counter[w] += 1
                            in_vocab_output_counts_in_epoch[w] += 1


            else:
                correct_answers_for_inaccurate_output_answers.append(epoch_labels[i])
                self.record_dict['correct_answers_for_inaccurate_output_answers'].append(epoch_labels[i])
                inaccurate_questions.append(epoch_questions[i])
                self.record_dict['inaccurate_questions'].append(epoch_questions[i])


                inaccurate_output_answers.append(epoch_outputs[i])
                self.record_dict['inaccurate_output_answers'].append(epoch_outputs[i])

                self.record_dict['incorrect_episode_ids'].append(episode_ids[i])
                incorrect_episode_ids.append(episode_ids[i])



                if self.task_name in tasks_outputing_non_yes_no_answers:
                    these_in_order_outputs = epoch_outputs[i].split()

                    these_in_order_labels = epoch_labels[i].split()

                    these_in_vocab_outputs = []

                    these_in_vocab_output_counts = Counter()

                    these_label_counts = Counter()

                    for w in these_in_order_outputs:
                        if w in dict_of_vocab_in_each_set['train']:
                            these_in_vocab_output_counts[w] += 1
                            in_vocab_output_counts_in_epoch[w] += 1

                    for w in these_in_order_labels:
                        these_label_counts[w] += 1
                        self.record_dict['label_words_in_epoch_counter'][w] += 1

                    extra_words_in_this_episode = these_in_vocab_output_counts - these_label_counts
                    missing_words_in_this_episode = these_label_counts - these_in_vocab_output_counts

                    accurate_words_in_this_episode = these_in_vocab_output_counts - extra_words_in_this_episode

                    extra_words_in_episodes_counter[episode_ids[i]] = extra_words_in_this_episode
                    missing_words_in_episodes_counter[episode_ids[i]] = missing_words_in_this_episode
                    accurate_words_in_episodes_counter[episode_ids[i]] = accurate_words_in_this_episode

                    self.record_dict['accurate_words_in_epoch_counter'] += accurate_words_in_this_episode

                    self.record_dict['missing_words_in_epoch_counter'] += missing_words_in_this_episode
                    self.record_dict['extra_words_in_epoch_counter'] += extra_words_in_this_episode
                    extra_words_in_epoch_counter += extra_words_in_this_episode
                    missing_words_in_epoch_counter += missing_words_in_this_episode

                    for w in missing_words_in_this_episode:
                        if w in action_words:
                            missing_actions_in_episodes_counter[episode_ids[i]][w] += 1
                            missing_actions_in_epoch_counter[w] += 1
                        elif w in prepositions:
                            pass

                        else:
                            missing_objects_in_episodes_counter[episode_ids[i]][w] += 1
                            missing_objects_in_epoch_counter[w] += 1

                    for w in extra_words_in_this_episode:
                        if w in action_words:
                            extra_actions_in_episodes_counter[episode_ids[i]][w] += 1
                            extra_actions_in_epoch_counter[w] += 1
                        elif w in prepositions:
                            pass
                        else:
                            extra_objects_in_episodes_counter[episode_ids[i]][w] += 1
                            extra_objects_in_epoch_counter[w] += 1



                    edit_distance = nltk.edit_distance(these_in_order_outputs, these_in_order_labels)
                    edit_distance_in_episodes[episode_ids[i]] = edit_distance
                    self.record_dict['edit_distances_in_epoch'].append(edit_distance)

                elif self.task_name in ['simple_object_yes_no', 'simple_action_yes_no', 'before_actions_yes_no', 
                'nat_lang_action_yes_no']:

                    these_in_order_questions= epoch_questions[i]
                    these_in_order_questions= these_in_order_questions.translate(str.maketrans('', '', string.punctuation))

                    these_in_order_questions = these_in_order_questions.split()



                    for w in these_in_order_questions:
                        if w in dict_of_vocab_in_each_set['train']:


                            self.record_dict['label_words_in_epoch_counter'][w] += 1

                            if epoch_labels[i] == 'yes':
                                self.record_dict['missing_words_in_epoch_counter'][w] += 1

                                missing_words_in_epoch_counter[w] += 1

                                if w in action_words:
                                    missing_actions_in_episodes_counter[episode_ids[i]][w] += 1
                                    missing_actions_in_epoch_counter[w] += 1

                                elif w in prepositions:
                                    pass

                                else:
                                    missing_objects_in_episodes_counter[episode_ids[i]][w] += 1
                                    missing_objects_in_epoch_counter[w] += 1

                            elif epoch_labels[i] == 'no':
                                self.record_dict['extra_words_in_epoch_counter'][w] += 1 

                                extra_words_in_epoch_counter[w] += 1

                                if w in action_words:
                                    extra_actions_in_episodes_counter[episode_ids[i]][w] += 1
                                    extra_actions_in_epoch_counter[w] += 1

                                elif w in prepositions:
                                    pass

                                else:
                                    extra_objects_in_episodes_counter[episode_ids[i]][w] += 1
                                    extra_objects_in_epoch_counter[w] += 1

                            else:
                                print("not yes or no")
                                IPython.embed()





        this_record_dict['accurate_questions'] = accurate_questions
        this_record_dict['accurate_output_answers'] = accurate_output_answers
        this_record_dict['inaccurate_questions'] = inaccurate_questions 
        this_record_dict['inaccurate_output_answers'] = inaccurate_output_answers
        this_record_dict['correct_episode_ids'] = correct_episode_ids
        this_record_dict['incorrect_episode_ids'] = incorrect_episode_ids
        this_record_dict['correct_answers_for_inaccurate_output_answers'] = correct_answers_for_inaccurate_output_answers






        this_record_dict['missing_actions_in_episodes'] = missing_actions_in_episodes_counter
        this_record_dict['missing_objects_in_episodes'] = missing_objects_in_episodes_counter
        this_record_dict['extra_actions_in_episodes'] = extra_actions_in_episodes_counter
        this_record_dict['extra_objects_in_episodes'] = extra_objects_in_episodes_counter
        this_record_dict['edit_distance_in_episodes'] = edit_distance_in_episodes


        this_record_dict['raw_output_by_id'] = raw_output_by_id

        self.record_dict['raw_output_by_id'].update(raw_output_by_id)

        self.record_dict['missing_actions_in_episodes'].update(missing_actions_in_episodes_counter)
        self.record_dict['missing_objects_in_episodes'].update(missing_objects_in_episodes_counter)
        self.record_dict['extra_actions_in_episodes'].update(extra_actions_in_episodes_counter)
        self.record_dict['extra_objects_in_episodes'].update(extra_objects_in_episodes_counter)
        self.record_dict['edit_distance_in_episodes'].update(edit_distance_in_episodes)





        this_record_dict['missing_words_in_epoch_counter'] = missing_words_in_epoch_counter
        this_record_dict['extra_words_in_epoch_counter'] = extra_words_in_epoch_counter


        this_record_dict['missing_actions_counter'] = missing_actions_in_epoch_counter
        this_record_dict['missing_objects_counter'] = missing_objects_in_epoch_counter


        self.record_dict['missing_objects_in_epoch_counter'] += missing_objects_in_epoch_counter
        self.record_dict['missing_actions_in_epoch_counter'] += missing_actions_in_epoch_counter
        self.record_dict['extra_actions_in_epoch_counter'] += extra_actions_in_epoch_counter
        self.record_dict['extra_objects_in_epoch_counter'] += extra_objects_in_epoch_counter

        this_record_dict['dataset_name'] = self.dataset_name
        this_record_dict['task_name'] = self.task_name
        this_record_dict['epoch'] = global_epoch

        num_examples = len(epoch_labels)
        acc = number_of_examples_correct / num_examples

        self.record_dict['number_of_examples_correct'] += number_of_examples_correct
        self.record_dict['total_number_of_examples'] += num_examples

        this_record_dict['acc'] = acc


        return this_record_dict


    def return_full_record_dict(self):
        self.record_dict['acc'] = self.record_dict['number_of_examples_correct'] / self.record_dict['total_number_of_examples']
        return self.record_dict

