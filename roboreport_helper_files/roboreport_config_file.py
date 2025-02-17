from argparse import Namespace
import pickle

args = Namespace(model_state_file="model.pth",
                 output_info_for_dir_name = 'alfred_test',
                 cuda=True,
                 device_num = 2,
                 device = 'cuda:2',
                 seed=333,
                 learning_rate=.0003,
                 number_of_epochs = 40,
                 batch_size=8,
                 valid_batch_size = 12,
                 num_batches_before_step = 2,
                 early_stopping_criteria=20,              
                 encoding_size=512,
                 num_workers = 0,
                 begin_seq_index = 1,
                 mask_index = 0,
                 num_layers = 3,
                 load_conv_layers = False,
                 use_sampling_for_negatives = True,
                 optimizer_type = 'adam',
                 save_best_model = False,
                 limit_dataset_size = False,
                 translate_pddl_to_english = True,
                 static_validate = False,
                 model_embed_size = 768,
                 model_checkpoint = 't5-base',
                 full_dataset = True,
                 use_trad_clip_feats = False,
                 use_clip_feats = True,
                 max_length = 1000,
                 load_pretrained_state_dict = False,
                 load_conv_layers_model = "",
                 main_task = "pddl_summ",
                 exclude_some_pddl = False
                 )