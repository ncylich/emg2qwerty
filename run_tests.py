import yaml
import subprocess
import os

# Base config file path
model_config_files = ["base.yaml", "conformer_decoder.yaml"]
training_data_configs = ["triple_user_half_data.yaml", "triple_user_quarter_data.yaml"]

for config_file in model_config_files:
    for training_data_config in training_data_configs:
        # Load the base configuration
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Modify configuration
        config["defaults", "user"] = training_data_config

        # Run training
        subprocess.run(
            ["python", "-m", "emg2qwerty.train", "trainer.devices=1", f"--config-name={config_file[:-5]}", "--multirun"]
        )

        print(f"Completed training {config_file} with {training_data_config}")

print("All training runs completed.")