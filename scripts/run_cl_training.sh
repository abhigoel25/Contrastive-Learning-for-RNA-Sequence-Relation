


# === Set main data directory once ===

# MAIN_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"
MAIN_DIR="/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"


# === Just specify file names ===
TRAIN_FILE="train_3primeIntron_filtered_min30views.pkl"
VAL_FILE="val_3primeIntron_filtered.pkl"
TEST_FILE="test_3primeIntron_filtered.pkl"

# TRAIN_FILE="train_merged_filtered_min30Views.pkl"
# VAL_FILE="val_merged_filtered_min30Views.pkl"
# TEST_FILE="test_merged_filtered_min30Views.pkl"

# TRAIN_FILE="train_5primeIntron_filtered.pkl"
# VAL_FILE="val_5primeIntron_filtered.pkl"
# TEST_FILE="test_5primeIntron_filtered.pkl"

# TRAIN_FILE="train_ExonSeq_filtered.pkl"
# VAL_FILE="val_ExonSeq_filtered.pkl"
# TEST_FILE="test_ExonSeq_filtered.pkl"

# TRAIN_FILE="ASCOT_data/train_ASCOT_merged_filtered_min30Views.pkl"
# VAL_FILE="ASCOT_data/val_ASCOT_merged_filtered_min30Views.pkl"
# TEST_FILE="ASCOT_data/test_ASCOT_merged_filtered_min30Views.pkl"

# TRAIN_FILE="ASCOT_data/train_ASCOT_3primeIntron_filtered.pkl"
# VAL_FILE="ASCOT_data/val_ASCOT_3primeIntron_filtered.pkl"
# TEST_FILE="ASCOT_data/test_ASCOT_3primeIntron_filtered.pkl"

# TRAIN_FILE="ASCOT_data/train_ASCOT_ExonSeq_filtered.pkl"
# VAL_FILE="ASCOT_data/val_ASCOT_ExonSeq_filtered.pkl"
# TEST_FILE="ASCOT_data/test_ASCOT_ExonSeq_filtered.pkl"

# # === Full paths constructed here ===
export TRAIN_DATA_FILE="${MAIN_DIR}/${TRAIN_FILE}"
export VAL_DATA_FILE="${MAIN_DIR}/${VAL_FILE}"
export TEST_DATA_FILE="${MAIN_DIR}/${TEST_FILE}"

# export CUDA_VISIBLE_DEVICES=1

NOTES="try"

# python -m scripts.cl_training \
#         task=introns_cl \
#         embedder="mtsplice"\
#         loss="weighted_supcon"\
#         tokenizer="onehot_tokenizer"\
#         task.global_batch_size=2048\
#         trainer.max_epochs=2 \
#         trainer.val_check_interval=1.0\
#         optimizer="sgd" \
#         trainer.devices=1\
#         logger.name="cl_trial_$(date +%Y%m%d_%H%M%S)"\
#         embedder.seq_len=400\
#         embedder.maxpooling=True\
#         logger.notes="$NOTES"\
#         dataset.n_augmentations=10 \
#         dataset.fixed_species=false\
#         dataset.train_data_file=$TRAIN_DATA_FILE \
#         dataset.val_data_file=$VAL_DATA_FILE \
#         dataset.test_data_file=$TEST_DATA_FILE

python -m scripts.cl_training \
        task=introns_cl \
        embedder="resnet"\
        loss="weighted_supcon"\
        tokenizer="custom_tokenizer"\
        task.global_batch_size=2048\
        trainer.max_epochs=2 \
        trainer.val_check_interval=1.0\
        optimizer="sgd" \
        trainer.devices=1\
        logger.name="cl_trial_$(date +%Y%m%d_%H%M%S)"\
        embedder.seq_len=201\
        embedder.maxpooling=True\
        logger.notes="$NOTES"\
        dataset.n_augmentations=3 \
        dataset.fixed_species=false\
        dataset.train_data_file=$TRAIN_DATA_FILE \
        dataset.val_data_file=$VAL_DATA_FILE \
        dataset.test_data_file=$TEST_DATA_FILE


       
# Directions:
# embedder: mtsplice, tokenizer: onehot_tokenizer, dataset: ...._merged_filtered_min30Views.pkl, tokenizer & embedder seqlen:400