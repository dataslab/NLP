CUDA_VISIBLE_DEVICES=3 python \
       Trainer.py --do_train --evaluate_during_training  \
                  --data_dir="data/dataset/NER/match" \
                  --output_dir="data/result/NER/match/bert_lstm_crf_4" \
                  --train_dataset="NER_train_4.json" \
                  --dev_dataset="NER_dev_4.json" \
                  --test_dataset="NER_test.json" \
                  --config_name="data/berts/chinese-macbert-base/config.json" \
                  --model_name_or_path="data/berts/chinese-macbert-base/pytorch_model.bin" \
                  --vocab_file="data/berts/chinese-macbert-base/vocab.txt" \
                  --word_vocab_file="/home/songhetian/dataset/ChineseEmbedding/tencent_vocab.txt" \
                  --max_scan_num=1000000 \
                  --max_word_num=5 \
                  --label_file="data/dataset/NER/match/labels.txt" \
                  --word_embedding="/home/songhetian/dataset/ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt" \
                  --saved_embedding_dir="data/dataset/NER/match/" \
                  --model_type="WCBertCRF_Token" \
                  --seed=1997 \
                  --per_gpu_train_batch_size=32 \
                  --per_gpu_eval_batch_size=16 \
                  --learning_rate=2e-3 \
                  --max_steps=-1 \
                  --max_seq_length=200 \
                  --num_train_epochs=15 \
                  --warmup_steps=190 \
                  --save_steps=600 \
                  --logging_steps=100 \
                #  --pgd \


#local rank是指定的gpu号
#记得改save_result的


