@echo off
set DATASET=data/carp_700m_test.jsonl
set CONFIG=configs/carp_700m.json
set TOKENIZER=artifacts/tokenizer_700m
set OUTPUT=artifacts/carp_700m_test_ckpt

python scripts/train_carp_sft.py ^
    --data %DATASET% ^
    --config %CONFIG% ^
    --tokenizer %TOKENIZER% ^
    --output-dir %OUTPUT% ^
    --reasoning-tokens 256 ^
    --block-size 512 ^
    --batch-size 1 ^
    --grad-accum-steps 4 ^
    --max-steps 10 ^
    --learning-rate 1e-4 ^
    --router-loss-weight 0.0 ^
    --ranking-loss-weight 0.5 ^
    --amp-dtype off ^
    --device cpu
