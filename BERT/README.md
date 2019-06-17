# BERT fine-tuning model
The pre-trained model is open source, which can be found on GitHub: https://github.com/google-research/bert.git. 
Only the class "DssbProcessor" in "run_classifier-1st-layer.py" and "run_classifier-2nd-layer.py" is modified.

The bert model 1 is trained to distinguish whether the sentence is evidence.
Put train15000-layer1-train.tsv, devset-layer1-eva.tsv to 1st-layer-train, then run the following script.
The model 1 will store under the output_dir.
--------------------------------------------------------------
python run_classifier-1st-layer.py \
  --task_name=dssb \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=1st-layer-train \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=1st-layer-outcome \
---------------------------------------------------------------

On the model 1, to predict whether the sentence is evidence or not, placing final-test-step1.tsv to 1st-layer-train.
Run the following script.
The predict outcome will show under the output_dir.
---------------------------------------------------------------
python run_classifier-1st-layer.py \
  --task_name=dssb \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=1st-layer-train \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=1st-layer-outcome \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=1st-layer-outcome \
---------------------------------------------------------------

The model 2 is trained to judge the fact of a claim.
Put train30000-layer2-train.tsv, devset-layer2-eva.tsv to 2nd-layer-train, then run the following script.
The model 2 will store under the output_dir.
---------------------------------------------------------------
!python run_classifier-2nd-layer.py \
  --task_name=dssb \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=2nd-layer-train \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=2nd-layer-outcome \
---------------------------------------------------------------

To preditic the label of a claim, putting final-test-step2.tsv to 2nd-layer-train.
Run the following script.
The predict outcome will show under the output_dir.
---------------------------------------------------------------
!python run_classifier-2nd-layer.py \
  --task_name=dssb \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=2nd-layer-train \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=2nd-layer-outcome \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=2nd-layer-outcome \
---------------------------------------------------------------