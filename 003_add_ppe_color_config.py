"""
Migration script to add PPE color configuration support
This allows different roles to have different colored helmets and vests
"""

from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_recognition"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def migrate():
    """Add PPE color configuration to database"""
    
    print("=" * 70)
    print("Adding PPE Color Configuration Support")
    print("=" * 70)
    
    # Check if color config already exists
    existing = db.system_config.find_one({"config_type": "ppe_color_rules"})
    if existing:
        print("[OK] ppe_color_rules config already exists -- skipping")
        return
    
    # Define color ranges in HSV format for better color detection
    # Format: [H_min, S_min, V_min, H_max, S_max, V_max]
    color_config = {
        "config_type": "ppe_color_rules",
        
        # Available colors with HSV ranges for detection
        "available_colors": {
            "yellow": {
                "name": "Yellow",
                "hsv_range": [[20, 100, 100], [30, 255, 255]],
                "display_color": "#FFFF00"
            },
            "orange": {
                "name": "Orange", 
                "hsv_range": [[10, 100, 100], [20, 255, 255]],
                "display_color": "#FF8800"
            },
            "red": {
                "name": "Red",
                "hsv_range": [[0, 100, 100], [10, 255, 255]],
                "display_color": "#FF0000"
            },
            "blue": {
                "name": "Blue",
                "hsv_range": [[100, 100, 100], [130, 255, 255]],
                "display_color": "#0000FF"
            },
            "green": {
                "name": "Green",
                "hsv_range": [[40, 100, 100], [80, 255, 255]],
                "display_color": "#00FF00"
            },
            "white": {
                "name": "White",
                "hsv_range": [[0, 0, 200], [180, 30, 255]],
                "display_color": "#FFFFFF"
            },
            "pink": {
                "name": "Pink",
                "hsv_range": [[140, 50, 100], [170, 255, 255]],
                "display_color": "#FF69B4"
            },
            "purple": {
                "name": "Purple",
                "hsv_range": [[130, 50, 50], [160, 255, 255]],
                "display_color": "#800080"
            }
        },
        
        # Role-based color requirements
        # Format: {"role": {"ppe_item": ["allowed_color1", "allowed_color2"]}}
        "role_color_requirements": {
            "worker": {
                "safety_helmet": ["yellow", "orange"],
                "reflective_vest": ["orange", "yellow"]
            },
            "engineer": {
                "safety_helmet": ["white", "blue"],
                "reflective_vest": ["orange", "yellow"]
            },
            "supervisor": {
                "safety_helmet": ["white"],
                "reflective_vest": ["orange", "yellow"]
            },
            "contractor": {
                "safety_helmet": ["yellow", "orange"],
                "reflective_vest": ["orange"]
            },
            "electrician": {
                "safety_helmet": ["yellow"],
                "reflective_vest": ["orange", "yellow"]
            },
            "welder": {
                "safety_helmet": ["red"],
                "reflective_vest": ["orange"]
            },
            "visitor": {
                "safety_helmet": ["blue", "white"],
                "reflective_vest": ["orange", "yellow"]
            },
            "manager": {
                "safety_helmet": ["white"],
                "reflective_vest": ["orange", "yellow"]
            },
            "default": {
                "safety_helmet": ["yellow", "orange", "white", "blue"],
                "reflective_vest": ["orange", "yellow"]
            }
        },
        
        # Enable/disable color checking
        "enable_color_checking": True,
        
        # Minimum color match percentage (0-100)
        "color_match_threshold": 30,
        
        "updated_at": datetime.utcnow(),
        "updated_by": "migration"
    }
    
    db.system_config.insert_one(color_config)
    print("[OK] PPE color configuration inserted successfully")
    print("\nColor Requirements by Role:")
    print("-" * 70)
    
    for role, requirements in color_config["role_color_requirements"].items():
        print(f"\n{role.upper()}:")
        for ppe_item, colors in requirements.items():
            print(f"  - {ppe_item}: {', '.join(colors)}")
    
    print("\n" + "=" * 70)
    print("[OK] Migration completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    migrate()
