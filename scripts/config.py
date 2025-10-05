from peft import LoraConfig

class Stage1Config:
    def __init__(self, r, lora_alpha, lora_dropout, learning_rate, num_train_epochs):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

class Stage2Config:
    def __init__(self, r, lora_alpha, lora_dropout, learning_rate, num_train_epochs):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

class ModelConfig:
    def __init__(self, stage1: Stage1Config, stage2: Stage2Config):
        self.stage1 = stage1
        self.stage2 = stage2

# --- Configurations for different model sizes ---

# Educated guesses for the 3B model
config_3b = ModelConfig(
    stage1=Stage1Config(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_train_epochs=6
    ),
    stage2=Stage2Config(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_train_epochs=4
    )
)

# Educated guesses for the 7B model
config_7b = ModelConfig(
    stage1=Stage1Config(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_train_epochs=6
    ),
    stage2=Stage2Config(
        r=64, # Increased rank for stage 2 on the 7B model
        lora_alpha=128,
        lora_dropout=0.1,
        learning_rate=5e-5, # Lower learning rate for the larger model in stage 2
        num_train_epochs=4
    )
)

def get_config(model_size: str):
    if "7b" in model_size.lower():
        print("--- Using configuration for 7B model. ---")
        return config_7b
    else:
        print("--- Using configuration for 3B model. ---")
        return config_3b
