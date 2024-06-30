
import sys
import json
import os
import pickle
from collections import defaultdict, Counter
import random
from copy import deepcopy
import IPython



def get_positive_and_negative_questions(place_one_items, actual_pddl):

    yes_questions = []
    no_questions = []

    split_lines = []

    for pddl_line in actual_pddl:
        split_line = pddl_line.split()
        split_lines.append(split_line)

    for pddl_line in actual_pddl:
        #changed to pddl_line from split_line
        if len(pddl_line) > 1:
            split_line = pddl_line.split()

            yes_questions.append(pddl_line)
            no_question = sample_negative_questions(split_line, split_lines, place_one_items )
            if no_question != None:

                no_question = " ".join(no_question)
                no_question = no_question.strip()
                no_questions.append(no_question)



    return yes_questions, no_questions




def get_positive_action_questions(actual_pddl):

    yes_questions = []

    split_lines = []


    for pddl_line in actual_pddl:
        if len(pddl_line) > 1:
            split_line = pddl_line.split()

            yes_questions.append(pddl_line)


    return yes_questions


def get_negative_action_questions(place_one_items, actual_pddl):

    no_questions = []

    split_lines = []

    for pddl_line in actual_pddl:
        split_line = pddl_line.split()
        split_lines.append(split_line)

    for pddl_line in actual_pddl:
        if len(pddl_line) > 1:
            split_line = pddl_line.split()

            no_question = sample_negative_questions(split_line, split_lines, place_one_items )
            if no_question != None:
                no_question = " ".join(no_question)
                no_question = no_question.strip()
                no_questions.append(no_question)



    return no_questions





def get_lists_of_objects_seen(all_events):


    all_objects_seen = []
    all_other_objects = set()

    for event in all_events:

        for obj in event['objects']:
            if obj['visible']:
                all_objects_seen.append(obj['objectType'])
            else: 
                all_other_objects.add(obj['objectType'])

    return all_objects_seen, all_other_objects



def make_negative_questions_about_objects_seen(objects_seen, all_objects_in_scene, all_possible_objects):

    negative_objects = deepcopy(all_possible_objects)

    for x in all_objects_in_scene:

        del negative_objects[x]


    return negative_objects


def make_right_before_questions(pddl):
    count = Counter()

    for x in pddl:
        count[x] += 1


    right_before_qs = []
    right_after_qs = []

    for x in range(1, len(pddl)):
        if count[pddl[x]] == 1:
            right_before_qs.append((pddl[x], pddl[x-1]))

    for x in range(len(pddl) - 1):
        if count[pddl[x]] == 1:
            right_after_qs.append((pddl[x], pddl[x+1]))

    return right_before_qs, right_after_qs




def make_nat_lang_right_before_questions(pddl, instructions):
    count = Counter()

    if len(pddl) != len(instructions):
        return None, None

    for x in pddl:
        count[x] += 1


    right_before_qs = []

    for x in range(1, len(pddl)):
        if count[pddl[x]] == 1:
            right_before_qs.append((instructions[x][:-1], pddl[x-1]))

    return right_before_qs



def make_nat_lang_right_after_questions(pddl, instructions):
    count = Counter()

    if len(pddl) != len(instructions):
        return None, None

    for x in pddl:
        count[x] += 1


    right_after_qs = []

    for x in range(len(pddl) - 1):
        if count[pddl[x]] == 1:
            right_after_qs.append((instructions[x][:-1], pddl[x+1]))

    return right_after_qs





def sample_negative_questions(pddl_line, full_pddl, place_one_items, number_of_samples_try=20):


    try:
        possible_completions = random.choices(list(place_one_items[pddl_line[0]].keys()), weights =
            list(place_one_items[pddl_line[0]].values()), k=number_of_samples_try)

    except:
        print("possible_completions first")
        IPython.embed()




    looking_for_negative = True


    indx = 0
    while looking_for_negative:



        negative_to_return = []
        negative_to_return.append(pddl_line[0])
        try:
            if type(possible_completions[indx]) == str:
                negative_to_return.append(possible_completions[indx])
            elif type(possible_completions[indx]) == tuple:
                negative_to_return.append(possible_completions[indx][0])
                negative_to_return.append(possible_completions[indx][1])
            else:
                print("neither str nor tuple")
                IPython.embed()

            if negative_to_return in full_pddl:
                indx += 1
            else:
                looking_for_negative = False
        except:
            print("possible_completions")
            negative_to_return = None
            looking_for_negative = False
            #IPython.embed()

    return negative_to_return


def generate_all_simple_negative_questions(pddl_line, full_pddl, place_one_items):



    first_word = pddl_line.split()[0]



    negative_qs = []

    for k,v in place_one_items[first_word].items():


        if type(k) == tuple:
            k = " ".join(k)

        this_action =  first_word + " " + k
        #IPython.embed()
        if this_action not in full_pddl:
            negative_qs.append((this_action, v))



    return negative_qs



def make_location_questions(episode):

    where_is_it_questions = []
    whats_on_top_questions = []


    objects_and_their_receptacles = defaultdict(set)
    receptacles_and_their_objects = defaultdict(set)

    for event in episode['events']:
        for obj in event['objects']:
            if obj['visible']:
                if obj['parentReceptacles']:
                    for x in obj['parentReceptacles']:
                        objects_and_their_receptacles[obj['objectType']].add(x.split("|")[0])


    for event in episode['events']:
        for obj in event['objects']:
            if obj['visible']:
                if obj['receptacleObjectIds']:
                    for x in obj['receptacleObjectIds']:
                        receptacles_and_their_objects[obj['objectType']].add(x.split("|")[0])





    #return where_is_it_questions, whats_on_top_questions
    return objects_and_their_receptacles, receptacles_and_their_objects

    
def make_all_temporal_order_questions(actual_pddl, padding = False, joining_text=None):
#questions all implicitly of the form is ITEM1 before ITEM2?

    padding_token = "PAD"
    
    yes_questions = []
    no_questions = []

    split_lines = []

    pddl_counts = defaultdict(int)

    for pddl_line in actual_pddl:
        pddl_counts[pddl_line] += 1


    len_this_pddl = len(actual_pddl)

    all_indices = list(range(len(actual_pddl)))



    for x in range(len(actual_pddl)-1):
        #if pddl_counts[actual_pddl[x]] > 1:
        #    continue
        #indices_to_try = random.sample(all_indices, len_this_pddl)

        #looking_for_match = True
        #found_forward = False
        #ffound_backward = False

        this_x_element = actual_pddl[x]
        """
        for y in range(x+1, len(actual_pddl)):
            this_y_element = actual_pddl[y]
            if this_y_element not in actual_pddl[:x]:
                if (this_x_element, this_y_element) not in yes_questions:
                    yes_questions.append((this_x_element, this_y_element))

        """

        for y in range(x+1, len(actual_pddl)):
            this_y_element = actual_pddl[y]
            yes_questions.append((this_x_element, this_y_element))




    reversed_y = []

    for x in yes_questions:
        reversed_y.append((x[1],x[0]))


    not_also_no_yes_qs = []

    for x in yes_questions:
        if x not in reversed_y:
            not_also_no_yes_qs.append(x)

    no_dup_yes_qs = []

    for x in not_also_no_yes_qs:
        if x not in no_dup_yes_qs:
            no_dup_yes_qs.append(x)


    no_dup_no_qs = []

    for x in no_dup_yes_qs:
        no_dup_no_qs.append((x[1],x[0]))




    """
    for x in range(len(actual_pddl)-1, 0, -1):
        #if pddl_counts[actual_pddl[x]] > 1:
        #    continue
        #indices_to_try = random.sample(all_indices, len_this_pddl)

        #looking_for_match = True
        #found_forward = False
        #ffound_backward = False

        this_x_element = actual_pddl[x]

        for y in range(x-1, -1, -1):
            this_y_element = actual_pddl[y]
            if this_y_element not in actual_pddl[x:]:
                if (this_x_element, this_y_element) not in no_questions:
                    no_questions.append((this_x_element, this_y_element))



        x_pddl = actual_pddl[x].split()
        i_pddl = actual_pddl[i].split()

        if padding:

            while len(x_pddl) < 3:
                x_pddl.append(padding_token)
            while len(i_pddl) < 3:
                i_pddl.append(padding_token)

        x_pddl = " ".join(x_pddl)
        i_pddl = " ".join(i_pddl)


        if joining_text:

            question = question_token.join([x_pddl, i_pddl])
        else:
            question = ((x_pddl, i_pddl))

        if i < x:
            no_questions.append(question)

        elif i > x:
            yes_questions.append(question)

        """

    return no_dup_yes_qs, no_dup_no_qs




def get_dict_of_action_counter(list_of_episodes):


    place_one_items = defaultdict(Counter)

    #dict_keys = list(dataset_dict.keys())

    for episode_dict in list_of_episodes:
        episode = episode_dict['clean_high_pddl']
        for pddl_line in episode:
            split_line = pddl_line.split()
            if len(split_line) == 2:
                place_one_items[split_line[0]][split_line[1]] += 1

            elif len(split_line) == 3:
                place_one_items[split_line[0]][(split_line[1], split_line[2])] += 1

    return place_one_items


"""
def get_low_pddl_dict_of_action_counter(list_of_episodes):


    place_one_items = defaultdict(Counter)

    #dict_keys = list(dataset_dict.keys())

    for episode_dict in list_of_episodes:
        episode = episode_dict['clean_low_pddl']
        for pddl_line in episode:
            split_line = pddl_line.split()
            if len(split_line) == 2:
                place_one_items[split_line[0]][split_line[1]] += 1

            elif len(split_line) == 3:
                place_one_items[split_line[0]][(split_line[1], split_line[2])] += 1

    return place_one_items

"""
