import json
import os

CONFIG_FILE = "config.json"

def loadConfig():
    if not os.path.exists(CONFIG_FILE):
        # Create default config if it doesn't exist
        default_config = {
            "path": os.path.join(os.getcwd(), "images"),
            "clip_model": "ViT-B-32",
            "ocr_languages": ["en"],
            "recursive_watch": True,
            "topK": 5
        }
        saveConfig(default_config)
        return default_config
    
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def saveConfig(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)