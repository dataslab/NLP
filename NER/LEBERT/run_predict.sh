CUDA_VISIBLE_DEVICES=0 python  \
       Trainer.py --do_predict   \
                  --data_dir="data/dataset/NER/match" \
                  --output_dir="data/result/NER/match/bert_lstm_crf_0" \
                  --train_dataset="NER_train_0.json" \
                  --dev_dataset="NER_dev_0.json" \
                  --test_dataset="NER_dev_0.json" \
                  --config_name="data/berts/chinese-macbert-base/config.json" \
                  --model_name_or_path="data/result/NER/match/bert_crf_0/checkpoint-1890/pytorch_model.bin" \
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
                  --max_seq_length=256 \
                  --num_train_epochs=20 \
                  --warmup_steps=190 \
                  --save_steps=600 \
                  --logging_steps=100 \



 #预测的模型用model_name_or_path加载注释
#   --model_name_or_path="data/result/NER/match/lebertcrf/checkpoint-3780" \