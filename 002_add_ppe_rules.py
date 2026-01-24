from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_recognition"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def migrate():
    existing = db.system_config.find_one({"config_type": "ppe_rules"})
    if existing:
        print("✅ ppe_rules config already exists — skipping")
        return

    doc = {
        "config_type": "ppe_rules",

        "available_ppe_classes": [
            "apron",
            "boots",
            "face_mask",
            "gloves",
            "hearing_muff",
            "reflective_vest",
            "safety_goggles",
            "safety_harness",
            "safety_helmet",
            "safety_jacket",
            "suit",
            "welding_mask"
        ],

        "role_rules": {
            "worker": [
                "safety_helmet",
                "reflective_vest",
                "gloves",
                "boots"
            ],
            "engineer": [
                "safety_helmet",
                "reflective_vest"
            ],
            "supervisor": [
                "safety_helmet",
                "reflective_vest"
            ],
            "contractor": [
                "safety_helmet",
                "reflective_vest",
                "gloves",
                "boots"
            ],
            "electrician": [
                "safety_helmet",
                "reflective_vest",
                "gloves",
                "safety_goggles"
            ],
            "welder": [
                "safety_helmet",
                "welding_mask",
                "gloves",
                "safety_jacket"
            ],
            "visitor": [
                "safety_helmet",
                "reflective_vest"
            ],
            "manager": [
                "safety_helmet"
            ],
            "default": [
                "safety_helmet",
                "reflective_vest"
            ]
        },

        "updated_at": datetime.utcnow(),
        "updated_by": "migration"
    }

    db.system_config.insert_one(doc)
    print("✅ ppe_rules config inserted successfully")

if __name__ == "__main__":
    migrate()
