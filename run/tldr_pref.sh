# Alignment

DATA_TYPE=p2r  # rw or p2r

# Any causal HuggingFace model (`AutoModelForCausalLM` class)
INIT_MODEL_NAME=pythia-2.8b

# local path to the SFT model, e.g., YOUR_PATH/models/pythia-2.8b_tldr_sft
INIT_MODEL_PATH=YOUR_INIT_MODEL_PATH

# local path to the training data, e.g., YOUR_PATH/exp/tldr_exp/data/tldr_${DATA_TYPE}
DATA_PATH=YOUR_DATA_PATH

# number of contrastive samples, should not be greater than the number of completion candidates in the dataset.
NUM_CONTRASTIVE=2

# compute
gpus=0,1,2,3
port=4349

# fpo arguments
ALPHA=0.025
betar=0.5
betapi=0.3
LOSS_TYPE=alphapo-pref


WANDB_API_KEY=YOUR_WANDB_API_KEY
WANDB_NAME=${LOSS_TYPE}

# Training
bash exp/tldr_exp/train_exo.sh $INIT_MODEL_NAME $INIT_MODEL_PATH $DATA_PATH $LOSS_TYPE $NUM_CONTRASTIVE ${WANDB_NAME}_train $ALPHA ${gpus} ${port} ${betar} ${betapi}
sleep 5

# Inference
bash exp/tldr_exp/inference_align.sh ${gpus} models/pythia-2.8b_tldr/align_${LOSS_TYPE}_nc${NUM_CONTRASTIVE} ${port}
sleep 5

# Evaluation
cd src

python api.py tldr ${NUM_CONTRASTIVE} ${WANDB_NAME}_sft sft ${LOSS_TYPE}
sleep 5

python api.py tldr ${NUM_CONTRASTIVE} ${WANDB_NAME}_chosen chosen ${LOSS_TYPE}

