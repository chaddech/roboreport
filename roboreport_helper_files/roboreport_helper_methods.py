import numpy as np
import torch
import random
import re
import pickle
from roboreport_helper_files.roboreport_config_file import args


if args.use_clip_feats:
    alfred_clip_file_name = "clip_rn50_last_layer_feat_conv.pt"
elif args.use_trad_clip_feats:
    alfred_clip_file_name = "traditional_clip_feats.pt"


prepositions = set(('in', 'on', 'up', 'to', 'the'))

vowels = ['a', 'i', 'e', 'o', 'u']

pddl_to_nl = {'gotolocation': 'go to the', 'pickupobject': 'pick up the', 'putobject': 'put the', 'coolobject': 'cool the',
    'heatobject': 'heat the', 'cleanobject': 'clean the', 'toggleobject': 'toggle the', 'sliceobject': 'slice the',
    'diningtable': 'dining table', 'sinkbasin': 'sink basin', 'sidetable': 'side table', 'butterknife': 'butter knife',
    'garbagecan': 'garbage can', 'tissuebox': 'tissue box', 'desklamp': 'desk lamp', 'winebottle': 'wine bottle', 
    'coffeetable': 'coffee table', 'spraybottle': 'spray bottle', 'floorlamp': 'floor lamp', 'alarmclock': 'alarm clock',
    'remotecontrol': 'remote control', 'coffeemachine': 'coffee machine', 'toiletpaper': 'toilet paper', 
    'toiletpaperhanger': 'toilet paper hanger', 'creditcard': 'credit card', 'stoveburner': 'stove burner', 
    'handtowelholder': 'hand towel holder', 'handtowel': 'hand towel', 'bathtubbasin': 'bathtub basin', 'soapbar':
    'soap bar', 'tennisracket': 'tennis racket', 'soapbottle': 'soap bottle', 'glassbottle': 'glass bottle', 'dishsponge': 
    'dish sponge', 'wateringcan': 'watering can', 'baseballbat': 'baseball bat', 'saltshaker': 'salt shaker',
    'peppershaker': 'pepper shaker', 'stoveknob': 'stove knob', 'showercurtain': 'shower curtain', 'tomatosliced':
    'sliced tomato', 'wateringcan': 'watering can', 'potatosliced': 'sliced potato', 'breadsliced': 'sliced bread',
    'applesliced': 'sliced apple', 'lettucesliced': 'sliced lettuce', 'eggcracked': 'cracked egg', 'laundryhamper': 
    'laundry hamper', 'laundryhamperlid': 'laundry hamper lid', 'tvstand': 'tv stand', 'footstool': 'foot stool',
    'showerhead': 'shower head', 'showerdoor': 'shower door', 'showerglass': 'shower glass', 'scrubbrush': 'scrub brush',
    'lightswitch': 'light switch', 'towlholder': 'towel holder'}


action_words = set(('gotolocation', 'pickupobject', 'putobject', 'coolobject', 'heatobject', 'cleanobject', 
'toggleobject', 'sliceobject', 'go', 'pick', 'put', 'cool', 'heat', 'clean', 'toggle', 'slice', 'moveahead', 
'rotateleft', 'rotateright', 'lookdown', 'lookup', 'closeobject', 'openobject'))


DICT_OF_VOCAB_IN_EACH_SET_FILE = 'lower_case_dict_of_vocab_in_each_set.pkl'
f = open(DICT_OF_VOCAB_IN_EACH_SET_FILE, 'rb')
dict_of_vocab_in_each_set = pickle.load(f)
f.close() 


places_that_take_in = set(('sinkbasin', 'bathtubbasin', 'cabinet', 'fridge', 'garbagecan', 'microwave', 'drawer'))

for w in prepositions:
    dict_of_vocab_in_each_set['train'].add(w)

for w in action_words:
    dict_of_vocab_in_each_set['train'].add(w)

for k,v in pddl_to_nl.items():
    for j in v.split():
        dict_of_vocab_in_each_set['train'].add(j)


def translate_pddl_to_english(pddl):


    new_text= []

    i = 0
    
    while i < len(pddl):
        w = pddl[i]
        if w in pddl_to_nl:

            new_text.append(pddl_to_nl[w])

            if w == 'putobject':

                i += 1

                w = pddl[i]
                if w in pddl_to_nl:
                    new_text.append(pddl_to_nl[w])
                else:
                    new_text.append(w)

                if pddl[i+1] in places_that_take_in:
                    new_text.append('in')
                else:
                    new_text.append('on')

                new_text.append('the')


        else:
            new_text.append(w)

        i += 1


    new_text_to_return = " ".join(new_text)

    new_text_to_return = re.sub(" , ", ", ", new_text_to_return)


    return new_text_to_return


# settings



TRAIN_DICT_FILE = "feat_dicts_processed_alfred_dataset/train_data.pkl"

VALID_SEEN_DICT_FILE = "feat_dicts_processed_alfred_dataset/valid_seen_data.pkl"

VALID_UNSEEN_DICT_FILE = "feat_dicts_processed_alfred_dataset/valid_unseen_data.pkl"

TRAIN_QUESTIONS_DICT_FILE_NAME = 'train_questions1.pkl'

VALID_SEEN_QUESTIONS_DICT_FILE_NAME = 'valid_seen_questions1.pkl'

VALID_UNSEEN_QUESTIONS_DICT_FILE_NAME = 'valid_unseen_questions1.pkl'

TRAIN_PDDL_FILE_NAME = 'lower_case_train_pddl.pkl'

VALID_SEEN_PDDL_FILE_NAME = 'lower_case_valid_seen_pddl.pkl'

VALID_UNSEEN_PDDL_FILE_NAME = 'lower_case_valid_unseen_pddl.pkl'

TRAIN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME = 'lower_case_train_simple_object_yes_nocorrected.pkl'

VALID_SEEN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME = 'lower_case_valid_seen_simple_object_yes_nocorrected.pkl'

VALID_UNSEEN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME = 'lower_case_valid_unseen_simple_object_yes_nocorrected.pkl'

TRAIN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME = 'lower_case_train_simple_action_yes_nocorrected.pkl'

VALID_SEEN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME = 'lower_case_valid_seen_simple_action_yes_nocorrected.pkl'

VALID_UNSEEN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME = 'lower_case_valid_unseen_simple_action_yes_nocorrected.pkl'

VALID_SEEN_ACTION_COUNTER_FILE = 'valid_seen_action_counter.pkl'

VALID_UNSEEN_ACTION_COUNTER_FILE = 'valid_unseen_action_counter.pkl'

TRAIN_ACTION_COUNTER_FILE = 'train_action_counter.pkl'


TRAIN_BEFORE_AFTER_QUESTIONS_FILE = 'lower_case_before_after_questions_train.pkl'

VALID_SEEN_BEFORE_AFTER_QUESTIONS_FILE = 'lower_case_before_after_questions_valid_seen.pkl'

VALID_UNSEEN_BEFORE_AFTER_QUESTIONS_FILE = 'lower_case_before_after_questions_valid_unseen.pkl'


#train ids to skip
train_ids_with_no_events = ['trial_T20190908_054316_003433',
 'trial_T20190909_062011_223446',
 'trial_T20190909_013637_168506',
 'trial_T20190907_222606_903630',
 'trial_T20190918_174139_904388',
 'trial_T20190909_053101_102010',
 'trial_T20190906_232604_097173',
 'trial_T20190907_185942_820847',
 'trial_T20190907_151643_465634',
 'trial_T20190908_142153_073870',
 'trial_T20190907_034714_802572']

train_ids_diff_len_pddl_instructions = ['trial_T20190906_181830_873214',
 'trial_T20190907_013546_073160',
 'trial_T20190907_013704_727644',
 'trial_T20190908_115507_503798',
 'trial_T20190908_120151_167011',
 'trial_T20190909_045706_358954']

train_ids_to_skip =  train_ids_with_no_events + train_ids_diff_len_pddl_instructions

valid_unseen_ids_to_skip = ['trial_T20190907_061009_396474']



#may want to skip training examples with exact same PDDL as validation set
overlapping_train_ids_file = 'train_ids_with_pddl_overlap_with_valid.pkl'
f = open(overlapping_train_ids_file, 'rb')
train_ids_with_pddl_overlap_with_valid = pickle.load(f)
f.close()
#train_ids_to_skip = train_ids_to_skip + train_ids_with_pddl_overlap_with_valid

"""
allowed_subset_of_train_ids_file = ALLOWED_SUBSET_FILE
f = open(allowed_subset_of_train_ids_file, "rb")
allowed_subset_of_train_ids = pickle.load(f)
f.close()

second_allowed_subset_of_train_ids_file = SECOND_ALLOWED_SUBSET_FILE
f = open(second_allowed_subset_of_train_ids_file, "rb")
second_allowed_subset_of_train_ids = pickle.load(f)
f.close()
"""

QUESTION_PREFIX = 'Was there a '


valid_seen_dicts = [VALID_SEEN_DICT_FILE, VALID_SEEN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME,
VALID_SEEN_PDDL_FILE_NAME, VALID_SEEN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME, VALID_SEEN_BEFORE_AFTER_QUESTIONS_FILE]

valid_unseen_dicts = [VALID_UNSEEN_DICT_FILE, VALID_UNSEEN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME,
VALID_UNSEEN_PDDL_FILE_NAME, VALID_UNSEEN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME, VALID_UNSEEN_BEFORE_AFTER_QUESTIONS_FILE]

train_dicts = [TRAIN_DICT_FILE, TRAIN_YES_N0_OBJECT_QUESTIONS_DICT_FILE_NAME,
TRAIN_PDDL_FILE_NAME, TRAIN_YES_N0_ACTION_QUESTIONS_DICT_FILE_NAME, TRAIN_BEFORE_AFTER_QUESTIONS_FILE]

f = open(VALID_SEEN_ACTION_COUNTER_FILE, 'rb')
valid_seen_action_counter = pickle.load(f)
f.close()


f = open(VALID_UNSEEN_ACTION_COUNTER_FILE, 'rb')
valid_unseen_action_counter = pickle.load(f)
f.close()

f = open(TRAIN_ACTION_COUNTER_FILE, 'rb')
train_action_counter = pickle.load(f)
f.close()