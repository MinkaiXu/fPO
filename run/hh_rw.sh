# Alignment

DATA_TYPE=rw  # rw or p2r

# Any causal HuggingFace model (`AutoModelForCausalLM` class)
INIT_MODEL_NAME=pythia-2.8b

# local path to the SFT model, e.g., YOUR_PATH/models/pythia-2.8b_hh/sft
INIT_MODEL_PATH=YOUR_INIT_MODEL_PATH

# local path to the training data, e.g., YOUR_PATH/exp/hh_exp/data/hh_${DATA_TYPE}
DATA_PATH=YOUR_DATA_PATH

# number of contrastive samples, should not be greater than the number of completion candidates in the dataset.
NUM_CONTRASTIVE=4

# compute
gpus=0,1,2,3
port=4349

# fpo arguments
ALPHA=0.1
betar=0.1
betapi=0.35
LOSS_TYPE=alphapo-rw


WANDB_API_KEY=YOUR_WANDB_API_KEY
WANDB_NAME=${LOSS_TYPE}

# Training
bash exp/hh_exp/train_exo.sh $INIT_MODEL_NAME $INIT_MODEL_PATH $DATA_PATH $LOSS_TYPE $NUM_CONTRASTIVE ${WANDB_NAME}_train $ALPHA ${gpus} ${port} ${betar} ${betapi}
sleep 5

# Inference
bash exp/hh_exp/inference_align.sh ${gpus} models/pythia-2.8b_hh/align_${LOSS_TYPE}_nc${NUM_CONTRASTIVE} ${port}
sleep 5

# Evaluation
cd src

python api.py hh ${NUM_CONTRASTIVE} ${WANDB_NAME}_sft sft ${LOSS_TYPE}
sleep 5

python api.py hh ${NUM_CONTRASTIVE} ${WANDB_NAME}_chosen chosen ${LOSS_TYPE}

