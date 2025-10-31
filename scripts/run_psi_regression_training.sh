# "intronexon"  # or "3p", "5p", "intronOnly"
# aux_models.5p_weights: "exprmnt_2025_06_08__20_39_37"
# aux_models.3p_weights: "exprmnt_2025_06_08__20_38_28"
NOTES="mtspliceBCEresnet variable"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        trainer.max_epochs=10\
        trainer.val_check_interval=0.5\
        embedder="resnet"\
        tokenizer="custom_tokenizer"\
        embedder.maxpooling=true\
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=true\
        aux_models.mtsplice_weights="exprmnt_2025_07_30__13_10_26"\
        aux_models.mode="intronexon"\
        aux_models.mtsplice_BCE=1\
        logger.name="Psi_VE_mtspliceBCEResnetFrz0Wrm0__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"
       
         
        