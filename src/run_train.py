import os

command = 'python run_classification_adv.py ' \
  '--task_name mr-adv ' \
  '--max_seq_len 128 ' \
  '--do_train ' \
  '--do_eval ' \
  '--attention 2 ' \
  '--learning_rate 3e-5 ' \
  '--data_dir data/MR/original_data ' \
  '--output_dir experiments/mr-roberta/ ' \
  '--model_name_or_path roberta-base ' \
  '--per_device_train_batch_size 16 ' \
  '--per_device_eval_batch_size 16 ' \
  '--num_train_epochs 5 ' \
  '--svd_reserve_size 0 ' \
  '--evaluation_strategy epoch ' \
  '--save_steps 1000000 '

os.system(command)


