"""
Hierarchical Scene Segmentation Pipeline
=========================================
Uses Grounding DINO for open-vocabulary object detection and SAM v1 for
precise mask segmentation. Recurses as deep as possible into detected objects
to build a full scene hierarchy tree.

Usage:
    python hierarchical_segment.py --images_dir /path/to/images

    python hierarchical_segment.py \
        --images_dir /path/to/images \
        --output_dir /path/to/output \
        --max_depth 4 \
        --box_threshold 0.30 \
        --text_threshold 0.25 \
        --min_region_px 1000 \
        --sam_checkpoint /path/to/sam_vit_h.pth \
        --gdino_model IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import tyro
from PIL import Image

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Prompt Hierarchy
# Defines what sub-objects to look for *inside* each detected category.
# Add or extend entries freely.
# ---------------------------------------------------------------------------
PROMPT_HIERARCHY: Dict[str, Optional[List[str]]] = {
    # =========================================================================
    # ROOT — every physically real thing we might encounter in any scene
    # =========================================================================
    "__root__": [
        # ---- architecture & urban infrastructure ----------------------------
        "building", "skyscraper", "tower", "apartment block", "house",
        "office building", "warehouse", "factory", "church", "mosque",
        "temple", "stadium", "library", "hospital", "school", "mall",
        "parking garage", "crane", "scaffolding", "construction site",
        "bridge", "overpass", "tunnel", "underpass", "dam", "lighthouse",
        "fence", "wall", "barrier", "railing", "gate", "pillar", "arch",
        "roof", "chimney", "antenna", "satellite dish", "solar panel",
        "water tower", "silo", "storage tank", "cooling tower",
        "fire escape", "staircase", "escalator", "elevator", "ramp",
        "road", "highway", "sidewalk", "crosswalk", "pavement",
        "parking lot", "road marking", "speed bump", "manhole cover",
        "curb", "median strip",
        # ---- vehicles -------------------------------------------------------
        "car", "truck", "bus", "van", "motorcycle", "bicycle",
        "scooter", "ambulance", "police car", "fire truck", "taxi",
        "pickup truck", "SUV", "sports car", "convertible",
        "tram", "subway car", "train", "locomotive", "cargo ship",
        "sailboat", "yacht", "speedboat", "kayak", "canoe",
        "airplane", "helicopter", "drone", "hot air balloon",
        "forklift", "excavator", "bulldozer", "tractor",
        "concrete mixer", "garbage truck", "delivery truck",
        "bicycle", "electric scooter", "skateboard", "roller skate",
        # ---- street furniture & signage ------------------------------------
        "traffic light", "traffic sign", "street sign", "stop sign",
        "yield sign", "no parking sign", "speed limit sign",
        "billboard", "advertisement board", "poster", "banner",
        "bus stop shelter", "bench", "picnic table",
        "lamp post", "street light", "fire hydrant",
        "parking meter", "trash can", "recycling bin", "dumpster",
        "mailbox", "telephone booth", "ATM", "vending machine",
        "newspaper stand", "flower stand", "street kiosk",
        "utility pole", "power line", "transformer box",
        # ---- people ---------------------------------------------------------
        "person", "pedestrian", "cyclist", "construction worker",
        "police officer", "security guard", "delivery person",
        "child", "elderly person",
        # ---- nature & vegetation -------------------------------------------
        "tree", "bush", "shrub", "grass", "flower", "plant",
        "flower pot", "planter box", "garden bed",
        "rock", "stone wall", "cliff", "mountain",
        "river", "lake", "pond", "fountain", "swimming pool",
        "fire", "smoke",
        # ---- animals -------------------------------------------------------
        "dog", "cat", "bird", "pigeon", "seagull", "horse",
        "cow", "sheep", "squirrel", "rat",
        # ---- outdoor furniture & recreation --------------------------------
        "playground equipment", "swing", "slide", "seesaw",
        "basketball hoop", "goal post", "tennis court", "sports field",
        "outdoor umbrella", "awning", "canopy", "tent",
        "statue", "sculpture", "monument",
        # ---- indoor / residential furniture --------------------------------
        "sofa", "couch", "armchair", "recliner",
        "dining table", "coffee table", "side table", "desk",
        "chair", "office chair", "bar stool", "bench seat",
        "bed", "bunk bed", "mattress", "pillow", "blanket",
        "wardrobe", "closet", "dresser", "nightstand",
        "bookshelf", "bookcase", "cabinet", "filing cabinet",
        "TV stand", "shelving unit", "storage rack",
        "lamp", "floor lamp", "ceiling light", "chandelier",
        "mirror", "picture frame", "painting", "wall art",
        "curtain", "blind", "window shade", "rug", "carpet",
        "door mat", "staircase railing", "indoor plant",
        "fireplace", "air conditioner", "radiator", "fan",
        # ---- kitchen & appliances ------------------------------------------
        "refrigerator", "freezer", "microwave", "oven", "stove",
        "dishwasher", "washing machine", "dryer", "vacuum cleaner",
        "toaster", "toaster oven", "blender", "coffee maker",
        "kettle", "rice cooker", "air fryer", "food processor",
        "kitchen sink", "faucet", "kitchen counter", "kitchen cabinet",
        "cutting board", "knife block", "spice rack",
        # ---- dining & food items -------------------------------------------
        "plate", "bowl", "mug", "cup", "glass", "wine glass",
        "fork", "knife", "spoon", "chopsticks", "spatula", "ladle",
        "pot", "pan", "wok", "baking tray", "mixing bowl",
        "food container", "lunch box", "thermos", "water bottle",
        "apple", "banana", "orange", "lemon", "grapes", "strawberry",
        "watermelon", "pineapple", "mango", "avocado", "tomato",
        "carrot", "broccoli", "lettuce", "onion", "potato",
        "bread", "baguette", "pizza", "burger", "sandwich",
        "cake", "cupcake", "donut", "cookie", "chocolate",
        "cereal box", "milk carton", "juice bottle", "soda can",
        "coffee cup", "tea cup", "fast food bag", "takeout container",
        # ---- grocery / retail items ----------------------------------------
        "shopping cart", "shopping basket", "grocery bag",
        "cardboard box", "plastic bag", "paper bag",
        "bottle", "can", "jar", "tin", "packet", "sachet",
        "cereal box", "detergent bottle", "shampoo bottle",
        "medicine bottle", "pill bottle", "first aid kit",
        "fire extinguisher", "smoke detector", "security camera",
        # ---- electronics & devices -----------------------------------------
        "laptop", "desktop computer", "monitor", "keyboard", "mouse",
        "tablet", "smartphone", "smartwatch", "earphones", "headphones",
        "speaker", "smart speaker", "television", "projector",
        "remote control", "game controller", "gaming console",
        "camera", "DSLR camera", "video camera", "webcam",
        "printer", "scanner", "router", "modem", "external hard drive",
        "USB hub", "power bank", "charger", "power strip", "extension cord",
        "calculator", "digital clock", "alarm clock",
        "microphone", "headset",
        # ---- office supplies -----------------------------------------------
        "pen", "pencil", "marker", "highlighter", "eraser",
        "ruler", "scissors", "tape", "stapler", "staple remover",
        "paper clip", "binder clip", "rubber band",
        "notebook", "notepad", "sticky note", "legal pad",
        "folder", "binder", "file organizer", "clipboard",
        "envelope", "stamp", "letter",
        "whiteboard", "blackboard", "chalkboard", "corkboard",
        "whiteboard marker", "chalk", "eraser board",
        "desk organizer", "paper tray", "inbox tray",
        "tape dispenser", "glue stick", "correction fluid",
        "sharpener", "hole punch", "laminator",
        # ---- bags & luggage ------------------------------------------------
        "backpack", "handbag", "shoulder bag", "tote bag",
        "briefcase", "laptop bag", "messenger bag",
        "suitcase", "travel bag", "duffel bag", "fanny pack",
        "wallet", "purse",
        # ---- clothing & accessories ----------------------------------------
        "shirt", "t-shirt", "jacket", "coat", "hoodie", "sweater",
        "pants", "jeans", "shorts", "skirt", "dress",
        "shoes", "sneakers", "boots", "sandals", "heels",
        "hat", "cap", "helmet", "beanie", "scarf", "gloves",
        "sunglasses", "glasses", "watch", "necklace", "bracelet",
        "tie", "bow tie", "belt", "umbrella",
        # ---- tools & hardware ----------------------------------------------
        "hammer", "screwdriver", "wrench", "pliers", "drill",
        "saw", "tape measure", "level", "chisel", "crowbar",
        "ladder", "toolbox", "tool belt", "safety helmet",
        "paint brush", "paint roller", "paint can",
        "cable", "rope", "chain", "lock", "padlock",
        # ---- medical & hygiene ---------------------------------------------
        "wheelchair", "crutches", "walking stick", "stretcher",
        "hospital bed", "IV stand", "stethoscope",
        "tissue box", "toilet paper", "soap dispenser", "hand sanitizer",
        "toothbrush", "toothpaste", "razor", "hairdryer",
        # ---- sports & recreation -------------------------------------------
        "ball", "football", "basketball", "tennis ball", "baseball",
        "soccer ball", "volleyball", "rugby ball",
        "tennis racket", "baseball bat", "golf club", "hockey stick",
        "bicycle helmet", "knee pad", "elbow pad",
        "yoga mat", "dumbbell", "barbell", "kettlebell", "treadmill",
        "surfboard", "snowboard", "ski", "roller skate",
        "fishing rod", "camping tent", "sleeping bag",
        # ---- toys & games --------------------------------------------------
        "toy", "doll", "action figure", "toy car", "toy truck",
        "building blocks", "lego", "puzzle", "board game",
        "playing cards", "chess board", "jigsaw puzzle",
        "stuffed animal", "teddy bear", "plush toy",
        "remote control car", "kite",
        # ---- books & media -------------------------------------------------
        "book", "magazine", "newspaper", "comic book",
        "textbook", "dictionary", "novel",
        "CD", "DVD", "vinyl record",
        # ---- art & craft ---------------------------------------------------
        "canvas", "easel", "paint tube", "paintbrush",
        "sketchpad", "colored pencil", "crayon",
        "clay", "pottery", "vase", "sculpture",
        # ---- musical instruments -------------------------------------------
        "guitar", "piano", "keyboard instrument", "violin", "cello",
        "drums", "drum kit", "saxophone", "trumpet", "flute",
        "microphone stand",
        # ---- signage & branding --------------------------------------------
        "logo", "brand sign", "neon sign", "storefront sign",
        "exit sign", "emergency sign", "information board",
        "menu board", "price tag", "barcode", "QR code",
        # ---- miscellaneous everyday objects --------------------------------
        "clock", "wall clock", "hourglass",
        "candle", "flower vase", "picture frame",
        "trash bag", "recycling bin", "compost bin",
        "fire extinguisher", "sprinkler system",
        "key", "lock", "doorknob",
        "coin", "banknote", "credit card",
        "sunscreen bottle", "lotion bottle",
        "gift box", "gift bag", "ribbon bow",
        "flag", "national flag", "pennant",
        # ---- household & indoor everyday ------------------------------------
        "laundry basket", "laundry hamper", "laundry bag",
        "clothes", "folded clothes", "pile of clothes",
        "paper towel roll", "paper towel", "paper towel holder",
        "toilet paper roll", "tissue roll",
        "thermostat", "wall thermostat", "smart thermostat", "temperature controller",
        "light switch", "wall switch", "switch plate", "light switch cover",
        "power outlet", "wall outlet", "electrical outlet", "wall socket",
        "outlet cover", "wall plate",
        "wire basket", "mesh basket", "fruit basket", "wire mesh basket",
        "produce basket", "wicker basket", "storage basket",
        "gallon jug", "water jug", "water gallon", "large water bottle",
        "spray bottle", "cleaning spray", "cleaning bottle", "spray can",
        "document", "paper", "sheet of paper", "receipt", "letter", "flyer",
        "plastic wrap", "cling film", "plastic bag of food",
        "kitchen island", "kitchen countertop", "countertop",
        "door", "interior door", "white door", "wooden door", "sliding door",
        "hallway", "corridor",
        "stool", "plastic stool", "plastic chair", "folding chair",
        "canister", "food canister", "storage canister", "container lid",
        "wall charger", "power adapter", "charging block", "plug",
        "floor mat", "bath mat", "welcome mat",
        "hardwood floor", "tile floor", "linoleum floor", "vinyl floor",
        "baseboard", "wall trim", "ceiling",
        "nail", "screw", "wall hook", "key hook",
        "hand soap", "dish soap", "cleaning product", "household cleaner",
        "rubber glove", "dish rag", "sponge", "scrub brush",
        "trash liner", "bin bag",
        "ziploc bag", "storage bag", "freezer bag",
        "foil", "aluminum foil", "plastic wrap roll",
        "paper plate", "paper cup", "disposable cup",
        "straw", "plastic straw",
        "napkin", "napkin holder", "paper napkin",
        "tablecloth", "placemat", "coaster",
        "food wrap", "food packaging", "snack bag", "chip bag",
        "condiment bottle", "ketchup bottle", "mustard bottle",
        "salt shaker", "pepper shaker", "spice bottle",
        "oil bottle", "vinegar bottle", "sauce bottle",
        "sugar container", "flour container",
        "magnets", "refrigerator magnet",
        "hook", "door stopper", "door wedge",
        "power strip", "surge protector",
        "smoke alarm", "carbon monoxide detector",
        "vent", "air vent", "heating vent", "floor vent",
        "window sill", "window ledge",
        "pipe", "plumbing pipe", "drain pipe",
    ],

    # =========================================================================
    # ARCHITECTURE
    # =========================================================================
    "building": [
        "window", "door", "balcony", "facade", "column", "arch",
        "entrance", "fire escape", "air conditioning unit", "antenna",
        "chimney", "roof", "cornice", "parapet", "awning",
        "windowsill", "gutter", "downspout", "graffiti",
        "signage", "logo", "security camera", "intercom",
    ],
    "skyscraper": [
        "window", "door", "balcony", "facade", "curtain wall",
        "rooftop", "entrance", "lobby entrance", "antenna",
        "logo", "signage",
    ],
    "apartment block": [
        "window", "door", "balcony", "facade", "entrance",
        "air conditioning unit", "fire escape", "mailbox",
    ],
    "tower": [
        "window", "door", "observation deck", "antenna", "elevator shaft",
    ],
    "house": [
        "window", "door", "roof", "chimney", "garage door",
        "porch", "fence", "garden", "mailbox", "driveway",
    ],
    "facade": [
        "window", "door", "balcony", "column", "arch",
        "decorative element", "relief", "inscription", "logo", "signage",
    ],
    "roof": [
        "chimney", "antenna", "solar panel", "skylight", "HVAC unit",
        "water tank", "satellite dish", "parapet", "rooftop garden",
        "gutter", "downspout",
    ],
    "entrance": [
        "door", "step", "ramp", "awning", "intercom", "mailbox",
        "security camera", "doorbell", "door handle",
    ],
    "balcony": [
        "railing", "door", "plant", "chair", "table", "laundry",
        "flower pot",
    ],
    "window": [
        "window frame", "glass pane", "windowsill", "shutter",
        "curtain", "blind", "mullion", "transom", "window latch",
    ],
    "door": [
        "door handle", "door panel", "door frame", "step", "intercom",
        "doorbell", "mail slot", "knocker", "lock", "hinge",
    ],
    "staircase": [
        "step", "handrail", "railing", "landing", "newel post",
    ],

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    "crane": [
        "jib", "tower mast", "hook", "counterweight",
        "operator cabin", "trolley", "hoist rope", "base",
    ],
    "scaffolding": [
        "platform", "support pole", "cross brace", "guardrail",
        "ladder", "mesh netting", "wheel",
    ],
    "construction site": [
        "crane", "scaffolding", "excavator", "bulldozer",
        "pile of construction material", "concrete mixer",
        "safety barrier", "construction worker", "hard hat",
        "toolbox", "ladder", "safety sign",
    ],
    "excavator": [
        "bucket", "arm", "cab", "tracks", "boom",
    ],

    # =========================================================================
    # VEHICLES
    # =========================================================================
    "car": [
        "wheel", "windshield", "side window", "door",
        "hood", "trunk", "side mirror", "headlight", "taillight",
        "bumper", "license plate", "roof rack", "wiper",
        "door handle", "seat", "steering wheel", "dashboard",
    ],
    "truck": [
        "wheel", "windshield", "cab door", "cargo area",
        "bumper", "side mirror", "headlight", "taillight",
        "exhaust pipe", "fuel tank", "license plate",
    ],
    "bus": [
        "wheel", "windshield", "passenger window", "door",
        "destination board", "headlight", "taillight",
        "bumper", "roof vent", "handrail", "seat",
    ],
    "van": [
        "wheel", "windshield", "side window", "door",
        "headlight", "taillight", "bumper", "sliding door",
        "license plate",
    ],
    "motorcycle": [
        "wheel", "handlebar", "seat", "engine", "exhaust pipe",
        "headlight", "taillight", "mirror", "fuel tank",
    ],
    "bicycle": [
        "wheel", "handlebar", "saddle", "pedal", "chain",
        "brake", "fork", "frame", "basket", "light",
    ],
    "train": [
        "wheel", "window", "door", "headlight", "pantograph",
        "locomotive", "cargo car", "passenger car",
    ],
    "airplane": [
        "wing", "engine", "fuselage", "tail", "cockpit window",
        "landing gear", "door", "propeller",
    ],

    # =========================================================================
    # PEOPLE
    # =========================================================================
    "person": [
        "head", "face", "eye", "nose", "mouth", "ear", "hair",
        "neck", "shoulder", "arm", "hand", "finger",
        "torso", "chest", "waist", "leg", "knee", "foot",
        "shirt", "jacket", "pants", "shoes", "hat",
        "backpack", "bag", "glasses", "watch",
    ],
    "pedestrian": [
        "head", "body", "arm", "leg", "bag", "clothing", "shoes",
    ],
    "face": [
        "eye", "eyebrow", "nose", "mouth", "lip", "chin",
        "cheek", "forehead", "ear",
    ],

    # =========================================================================
    # STREET FURNITURE & SIGNAGE
    # =========================================================================
    "traffic light": [
        "red light", "green light", "yellow light",
        "pedestrian signal", "housing", "pole", "countdown timer",
    ],
    "lamp post": [
        "light fixture", "pole", "bracket", "bulb",
    ],
    "street sign": [
        "sign board", "text", "pole", "reflector",
    ],
    "billboard": [
        "advertisement", "frame", "support structure", "lighting",
        "logo", "text", "image",
    ],
    "vending machine": [
        "slot", "button", "display screen", "product window",
        "coin slot", "card reader",
    ],
    "ATM": [
        "screen", "keypad", "card slot", "cash slot",
        "camera", "receipt slot",
    ],
    "bus stop shelter": [
        "roof", "glass panel", "bench", "timetable board",
        "advertisement panel", "pole",
    ],

    # =========================================================================
    # NATURE
    # =========================================================================
    "tree": [
        "trunk", "branch", "foliage", "root", "bark",
        "leaf", "fruit", "seed",
    ],
    "flower": [
        "petal", "stem", "leaf", "stamen", "pistil",
    ],
    "flower pot": [
        "pot", "soil", "plant", "flower",
    ],
    "fountain": [
        "basin", "spout", "nozzle", "sculpture", "water jet",
    ],

    # =========================================================================
    # ANIMALS
    # =========================================================================
    "dog": [
        "head", "snout", "ear", "eye", "body", "leg", "paw",
        "tail", "collar", "leash",
    ],
    "cat": [
        "head", "ear", "eye", "whisker", "body", "paw", "tail",
        "collar",
    ],
    "bird": [
        "beak", "eye", "wing", "body", "tail feather", "claw",
    ],

    # =========================================================================
    # INDOOR FURNITURE
    # =========================================================================
    "sofa": [
        "cushion", "armrest", "backrest", "leg", "pillow",
    ],
    "bed": [
        "mattress", "pillow", "bed frame", "headboard", "footboard",
        "bedsheet", "blanket", "duvet",
    ],
    "desk": [
        "surface", "drawer", "leg", "monitor", "keyboard",
        "mouse", "lamp", "pen holder",
    ],
    "bookshelf": [
        "shelf", "book", "frame", "bracket",
    ],
    "chair": [
        "seat", "backrest", "armrest", "leg",
    ],
    "office chair": [
        "seat", "backrest", "armrest", "wheel base", "gas lift",
        "headrest",
    ],
    "dining table": [
        "table top", "leg", "plate", "bowl", "cup", "cutlery",
    ],
    "cabinet": [
        "door", "handle", "shelf", "drawer", "hinge",
    ],
    "wardrobe": [
        "door", "handle", "shelf", "hanging rail", "drawer",
        "mirror",
    ],
    "mirror": [
        "glass", "frame", "backing",
    ],
    "lamp": [
        "shade", "bulb", "base", "cord", "switch",
    ],
    "fireplace": [
        "mantel", "firebox", "grate", "hearth", "flue",
        "fire", "logs",
    ],

    # =========================================================================
    # KITCHEN & APPLIANCES
    # =========================================================================
    "refrigerator": [
        "door", "handle", "shelf", "freezer compartment",
        "vegetable drawer", "ice maker", "door seal",
    ],
    "oven": [
        "door", "handle", "knob", "rack", "window",
        "baking element", "control panel",
    ],
    "stove": [
        "burner", "knob", "grate", "pan", "control panel",
    ],
    "microwave": [
        "door", "window", "turntable", "keypad", "handle",
    ],
    "dishwasher": [
        "door", "handle", "basket", "spray arm", "control panel",
    ],
    "washing machine": [
        "door", "drum", "detergent drawer", "control panel", "porthole",
    ],
    "coffee maker": [
        "carafe", "filter basket", "water reservoir", "control panel",
        "coffee pod slot", "nozzle",
    ],
    "toaster": [
        "slot", "lever", "crumb tray", "knob", "heating element",
    ],
    "kitchen sink": [
        "basin", "faucet", "drain", "handle", "soap dispenser",
    ],
    "faucet": [
        "spout", "handle", "valve", "aerator",
    ],
    "pot": [
        "body", "handle", "lid", "base",
    ],
    "pan": [
        "body", "handle", "base", "coating",
    ],

    # =========================================================================
    # DINING
    # =========================================================================
    "plate": [
        "rim", "well", "food",
    ],
    "cup": [
        "body", "handle", "rim", "base",
    ],
    "wine glass": [
        "bowl", "stem", "base", "rim",
    ],
    "knife": [
        "blade", "handle", "tip", "guard",
    ],
    "fork": [
        "tine", "handle", "neck",
    ],

    # =========================================================================
    # FOOD
    # =========================================================================
    "pizza": [
        "crust", "cheese", "topping", "sauce", "slice",
    ],
    "burger": [
        "bun", "patty", "lettuce", "tomato", "cheese", "sauce",
        "pickle", "onion",
    ],
    "sandwich": [
        "bread", "filling", "lettuce", "cheese", "meat",
    ],
    "cake": [
        "layer", "frosting", "decoration", "candle", "slice",
    ],
    "apple": [
        "skin", "stem", "leaf",
    ],
    "banana": [
        "peel", "flesh",
    ],

    # =========================================================================
    # ELECTRONICS
    # =========================================================================
    "laptop": [
        "screen", "keyboard", "touchpad", "hinge", "webcam",
        "port", "speaker", "battery indicator",
    ],
    "desktop computer": [
        "tower", "monitor", "keyboard", "mouse", "cable",
        "power button", "USB port", "fan vent",
    ],
    "monitor": [
        "screen", "stand", "bezel", "button", "cable port",
        "power indicator",
    ],
    "keyboard": [
        "key", "spacebar", "function key", "number pad",
        "USB cable", "indicator light",
    ],
    "mouse": [
        "left button", "right button", "scroll wheel",
        "body", "cable",
    ],
    "smartphone": [
        "screen", "camera", "speaker", "button",
        "port", "case", "SIM tray",
    ],
    "tablet": [
        "screen", "camera", "button", "speaker", "port", "case",
    ],
    "television": [
        "screen", "bezel", "stand", "remote sensor",
        "port", "speaker", "power button",
    ],
    "remote control": [
        "button", "battery compartment", "IR emitter", "screen",
    ],
    "camera": [
        "lens", "body", "viewfinder", "shutter button",
        "flash", "strap", "card slot", "battery door",
    ],
    "headphones": [
        "ear cup", "headband", "ear pad", "cable", "jack",
        "driver", "microphone",
    ],
    "speaker": [
        "grille", "driver", "body", "port", "volume knob",
        "power button", "cable",
    ],
    "printer": [
        "paper tray", "output tray", "cover", "control panel",
        "ink port", "USB port", "display",
    ],
    "router": [
        "antenna", "body", "indicator light", "port",
        "power button",
    ],
    "game controller": [
        "button", "joystick", "trigger", "d-pad",
        "bumper", "battery compartment", "cable",
    ],

    # =========================================================================
    # OFFICE SUPPLIES
    # =========================================================================
    "pen": [
        "cap", "barrel", "tip", "clip", "ink cartridge",
    ],
    "pencil": [
        "tip", "eraser", "ferrule", "barrel",
    ],
    "stapler": [
        "body", "staple tray", "anvil", "handle",
    ],
    "scissors": [
        "blade", "handle", "pivot screw",
    ],
    "notebook": [
        "cover", "page", "spine", "binding", "line",
    ],
    "folder": [
        "cover", "tab", "pocket", "label",
    ],
    "whiteboard": [
        "surface", "frame", "tray", "marker holder",
    ],
    "desk organizer": [
        "pen holder", "tray", "compartment", "drawer",
    ],
    "clipboard": [
        "board", "clip", "paper",
    ],

    # =========================================================================
    # BAGS & LUGGAGE
    # =========================================================================
    "backpack": [
        "main compartment", "pocket", "strap", "zipper",
        "handle", "clip", "mesh pocket",
    ],
    "handbag": [
        "strap", "body", "pocket", "zipper", "clasp",
        "handle", "logo",
    ],
    "suitcase": [
        "body", "handle", "wheel", "zipper", "lock",
        "strap", "corner guard",
    ],
    "briefcase": [
        "body", "handle", "latch", "strap", "compartment",
        "combination lock",
    ],

    # =========================================================================
    # CLOTHING & ACCESSORIES
    # =========================================================================
    "jacket": [
        "collar", "sleeve", "pocket", "zipper", "button",
        "lining", "lapel", "cuff", "hood",
    ],
    "shirt": [
        "collar", "sleeve", "button", "pocket", "cuff", "hem",
    ],
    "shoes": [
        "sole", "upper", "lace", "tongue", "heel", "toe cap",
        "insole",
    ],
    "hat": [
        "crown", "brim", "band", "button", "visor",
    ],
    "glasses": [
        "lens", "frame", "nose pad", "temple", "hinge",
    ],
    "watch": [
        "dial", "strap", "crown", "bezel", "hands",
        "case", "crystal",
    ],
    "umbrella": [
        "canopy", "shaft", "handle", "rib", "tip",
        "button", "wrist strap",
    ],

    # =========================================================================
    # TOOLS
    # =========================================================================
    "hammer": [
        "head", "handle", "claw", "face", "grip",
    ],
    "drill": [
        "body", "chuck", "bit", "handle", "trigger",
        "battery", "cord",
    ],
    "ladder": [
        "rung", "side rail", "foot", "locking mechanism",
    ],
    "toolbox": [
        "lid", "handle", "tray", "compartment", "latch",
        "hammer", "screwdriver", "wrench",
    ],

    # =========================================================================
    # SPORTS & RECREATIONAL
    # =========================================================================
    "bicycle helmet": [
        "shell", "foam", "strap", "buckle", "vent",
    ],
    "dumbbell": [
        "plate", "handle", "collar",
    ],
    "ball": [
        "surface", "seam", "valve",
    ],
    "yoga mat": [
        "surface", "grip texture", "strap",
    ],

    # =========================================================================
    # TOYS & GAMES
    # =========================================================================
    "doll": [
        "head", "face", "hair", "body", "arm", "leg",
        "clothing", "shoe",
    ],
    "toy car": [
        "body", "wheel", "window", "door", "hood",
    ],
    "board game": [
        "board", "dice", "piece", "card", "token", "box",
    ],
    "lego": [
        "brick", "baseplate", "figure", "connector",
    ],
    "teddy bear": [
        "head", "ear", "eye", "nose", "body", "arm", "leg",
        "paw", "bow",
    ],

    # =========================================================================
    # BOOKS & MEDIA
    # =========================================================================
    "book": [
        "cover", "spine", "page", "title", "author name",
        "bookmark", "illustration",
    ],
    "magazine": [
        "cover", "page", "headline", "advertisement", "photo",
    ],

    # =========================================================================
    # MUSICAL INSTRUMENTS
    # =========================================================================
    "guitar": [
        "body", "neck", "headstock", "tuning peg", "string",
        "fret", "bridge", "sound hole", "strap",
    ],
    "piano": [
        "key", "pedal", "lid", "bench", "music stand",
        "black key", "white key",
    ],
    "drums": [
        "snare drum", "bass drum", "tom", "cymbal", "hi-hat",
        "drum stick", "pedal", "drum stool",
    ],

    # =========================================================================
    # MEDICAL
    # =========================================================================
    "wheelchair": [
        "seat", "backrest", "wheel", "armrest", "footrest",
        "push handle", "brake",
    ],
    "first aid kit": [
        "bandage", "antiseptic", "scissors", "glove", "tape",
        "gauze", "instruction card",
    ],
    "fire extinguisher": [
        "cylinder", "handle", "nozzle", "pin", "pressure gauge",
        "label",
    ],

    # =========================================================================
    # SIGNAGE & BRANDING
    # =========================================================================
    "logo": [
        "symbol", "text", "icon", "brand name", "tagline",
    ],
    "exit sign": [
        "text", "arrow", "housing", "light",
    ],
    "neon sign": [
        "tube", "text", "frame", "support",
    ],
    "menu board": [
        "text", "price", "image", "section header", "frame",
    ],

    # =========================================================================
    # INFRASTRUCTURE
    # =========================================================================
    "bridge": [
        "deck", "pillar", "railing", "arch",
        "cable", "suspension tower", "abutment",
    ],
    "power line": [
        "wire", "insulator", "transmission tower",
    ],
    "solar panel": [
        "cell", "frame", "glass", "junction box", "mounting bracket",
    ],
    "security camera": [
        "lens", "housing", "mount", "cable", "IR emitter",
    ],
    "fire hydrant": [
        "valve", "cap", "body", "flange",
    ],
    "manhole cover": [
        "lid", "frame", "pattern", "handle",
    ],

    # =========================================================================
    # HOUSEHOLD & INDOOR
    # =========================================================================
    "laundry basket": [
        "clothes", "shirt", "pants", "towel", "sock",
        "handle", "lid",
    ],
    "laundry hamper": [
        "clothes", "handle", "lid",
    ],
    "wire basket": [
        "rim", "wire", "handle", "produce", "onion", "fruit",
    ],
    "fruit basket": [
        "apple", "banana", "orange", "onion", "potato", "fruit",
    ],
    "paper towel roll": [
        "sheet", "core", "perforations",
    ],
    "paper towel holder": [
        "rod", "base", "paper towel roll",
    ],
    "thermostat": [
        "display", "button", "temperature reading", "screen",
        "body", "sensor",
    ],
    "light switch": [
        "switch", "cover plate", "screw",
    ],
    "switch plate": [
        "switch", "outlet", "screw", "cover",
    ],
    "power outlet": [
        "socket", "cover plate", "screw", "ground hole",
    ],
    "wall outlet": [
        "socket", "cover plate", "screw",
    ],
    "spray bottle": [
        "trigger", "nozzle", "bottle body", "label", "liquid",
    ],
    "cleaning spray": [
        "trigger", "nozzle", "bottle body", "label",
    ],
    "gallon jug": [
        "cap", "handle", "body", "label", "liquid",
    ],
    "water jug": [
        "cap", "handle", "body", "label", "liquid",
    ],
    "canister": [
        "lid", "body", "label", "handle",
    ],
    "kitchen island": [
        "countertop", "cabinet door", "drawer", "handle",
        "bar stool", "chair",
    ],
    "countertop": [
        "edge", "surface", "backsplash",
    ],
    "door": [
        "door handle", "door panel", "door frame", "step", "intercom",
        "doorbell", "mail slot", "knocker", "lock", "hinge", "peephole",
    ],
    "interior door": [
        "door handle", "door panel", "door frame", "lock", "hinge",
    ],
    "stool": [
        "seat", "leg", "footrest",
    ],
    "plastic chair": [
        "seat", "backrest", "leg", "armrest",
    ],
    "folding chair": [
        "seat", "backrest", "frame", "hinge",
    ],
    "document": [
        "text", "heading", "table", "logo",
    ],
    "receipt": [
        "text", "logo", "barcode", "total amount",
    ],
    "hallway": [
        "door", "wall", "floor", "light", "rug",
    ],
    "power adapter": [
        "plug", "cable", "body", "port",
    ],
    "wall charger": [
        "plug prong", "body", "port", "LED indicator",
    ],
    "power strip": [
        "outlet", "switch", "cable", "indicator light",
    ],
    "air vent": [
        "grille", "slat", "frame", "damper",
    ],
    "floor vent": [
        "grille", "slat", "frame",
    ],
    "smoke alarm": [
        "body", "indicator light", "test button", "sensor",
    ],
    "condiment bottle": [
        "cap", "label", "body", "nozzle",
    ],
    "spice bottle": [
        "cap", "label", "body", "shaker top",
    ],
    "oil bottle": [
        "cap", "label", "body", "spout",
    ],
    "napkin holder": [
        "napkin", "body", "base",
    ],
    "tablecloth": [
        "fabric", "edge", "pattern",
    ],
    "placemat": [
        "surface", "edge",
    ],
    "coaster": [
        "surface", "edge",
    ],
    "trash bag": [
        "body", "tie", "knot",
    ],
    "ziploc bag": [
        "seal", "zipper", "body", "content",
    ],
    "chip bag": [
        "seal", "label", "body",
    ],
    "snack bag": [
        "seal", "label", "body",
    ],
    "sponge": [
        "scrub side", "soft side", "body",
    ],
    # ---- default: no further children -----------------------------------
    "__default__": None,
}


def get_child_prompts(label: str) -> Optional[List[str]]:
    """Return sub-prompts for a detected label, or None if leaf node."""
    label_lower = label.lower().strip()
    # Direct match
    if label_lower in PROMPT_HIERARCHY:
        return PROMPT_HIERARCHY[label_lower]
    # Partial / substring match
    for key in PROMPT_HIERARCHY:
        if key.startswith("__"):
            continue
        if key in label_lower or label_lower in key:
            result = PROMPT_HIERARCHY[key]
            if result is not None:
                return result
    return PROMPT_HIERARCHY.get("__default__", None)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detected and segmented object."""
    label: str
    score: float
    bbox_xyxy: List[int]          # absolute pixel coords in the *original* image
    mask: Optional[np.ndarray]    # H×W bool mask in the *original* image space
    depth: int                    # nesting depth (0 = scene-level)
    children: List["Detection"] = field(default_factory=list)
    node_id: str = ""             # unique id for this node

    def to_dict(self, include_mask: bool = False) -> dict:
        d = {
            "id": self.node_id,
            "label": self.label,
            "score": round(float(self.score), 4),
            "bbox_xyxy": self.bbox_xyxy,
            "depth": self.depth,
            "children": [c.to_dict(include_mask=include_mask) for c in self.children],
        }
        if include_mask and self.mask is not None:
            # RLE-style compact representation
            d["mask_shape"] = list(self.mask.shape)
        return d


# ---------------------------------------------------------------------------
# Grounding DINO wrapper
# ---------------------------------------------------------------------------

class GroundingDINODetector:
    """
    Wraps HuggingFace Grounding DINO for open-vocabulary detection.
    Supports both:
      - IDEA-Research/grounding-dino-tiny  (fast, lower accuracy)
      - IDEA-Research/grounding-dino-base  (recommended)
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        prompt_chunk_size: int = 20,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"[GroundingDINO] Loading model: {model_id}")
        self.device = device
        # Grounding DINO has a ~256-token limit on the text prompt.
        # We split the prompt list into chunks of this size and merge results.
        self.prompt_chunk_size = max(1, prompt_chunk_size)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        print(f"[GroundingDINO] Ready on {device}  (prompt_chunk_size={self.prompt_chunk_size})")

    def _detect_chunk(
        self,
        image: Image.Image,
        prompts: List[str],
        box_threshold: float,
        text_threshold: float,
    ) -> List[Tuple[str, float, List[int]]]:
        """Run detection for a single chunk of prompts (≤ prompt_chunk_size)."""
        text_prompt = " . ".join(prompts) + " ."
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # transformers ≥5.x uses a single `threshold`; ≤4.x uses two separate ones.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

        w, h = image.size
        dets: List[Tuple[str, float, List[int]]] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            dets.append((label, float(score), [x1, y1, x2, y2]))
        return dets

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        prompts: List[str],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> List[Tuple[str, float, List[int]]]:
        """
        Returns list of (label, score, [x1,y1,x2,y2]) in absolute pixel coords.
        Automatically chunks `prompts` into batches of `prompt_chunk_size` to
        stay within Grounding DINO's ~256-token text limit.
        """
        all_dets: List[Tuple[str, float, List[int]]] = []
        for i in range(0, len(prompts), self.prompt_chunk_size):
            chunk = prompts[i : i + self.prompt_chunk_size]
            all_dets.extend(
                self._detect_chunk(image, chunk, box_threshold, text_threshold)
            )
        return all_dets


# ---------------------------------------------------------------------------
# SAM v1 wrapper
# ---------------------------------------------------------------------------

class SAMSegmenter:
    """
    Wraps Meta's Segment Anything Model (SAM v1) using bounding-box prompts.
    Downloads the checkpoint automatically if not found.
    """

    CHECKPOINT_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda",
        cache_dir: str | Path = "~/.cache/sam",
    ):
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint is None:
            checkpoint = str(cache_dir / f"sam_{model_type}.pth")

        if not Path(checkpoint).exists():
            url = self.CHECKPOINT_URLS[model_type]
            print(f"[SAM] Downloading {model_type} checkpoint from {url} ...")
            import urllib.request
            urllib.request.urlretrieve(url, checkpoint, reporthook=self._dl_progress)
            print()

        print(f"[SAM] Loading {model_type} from {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        self.sam_model = sam
        self.predictor = SamPredictor(sam)
        self._auto_mask_generator_cls = SamAutomaticMaskGenerator
        self._auto_mask_generator = None
        self._auto_mask_cfg = None
        self._current_image_set = False
        print(f"[SAM] Ready on {device}")

    @staticmethod
    def _dl_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 5)
        sys.stdout.write(f"\r  [{bar:<20}] {pct}%")
        sys.stdout.flush()

    def set_image(self, image_rgb: np.ndarray):
        """Set the image for subsequent predict() calls (amortises encoder cost)."""
        self.predictor.set_image(image_rgb)
        self._current_image_set = True

    def predict_box(self, box_xyxy: List[int]) -> np.ndarray:
        """
        Returns a H×W boolean mask for the given bounding box.
        `set_image` must be called before this.
        """
        assert self._current_image_set, "Call set_image() first"
        box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
        # Pick the highest-scoring mask
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    def generate_masks(
        self,
        image_rgb: np.ndarray,
        min_mask_area: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        points_per_side: int,
        points_per_batch: int,
        crop_n_layers: int,
        crop_n_points_downscale_factor: int,
    ) -> List[dict]:
        """
        Class-agnostic segmentation proposals via SAM AutomaticMaskGenerator.
        Returns masks sorted by descending area.
        """
        cfg = (
            int(min_mask_area),
            float(pred_iou_thresh),
            float(stability_score_thresh),
            int(points_per_side),
            int(points_per_batch),
            int(crop_n_layers),
            int(crop_n_points_downscale_factor),
        )
        if self._auto_mask_generator is None or self._auto_mask_cfg != cfg:
            self._auto_mask_generator = self._auto_mask_generator_cls(
                model=self.sam_model,
                min_mask_region_area=int(min_mask_area),
                pred_iou_thresh=float(pred_iou_thresh),
                stability_score_thresh=float(stability_score_thresh),
                points_per_side=int(points_per_side),
                points_per_batch=int(points_per_batch),
                crop_n_layers=int(crop_n_layers),
                crop_n_points_downscale_factor=int(crop_n_points_downscale_factor),
            )
            self._auto_mask_cfg = cfg

        masks = self._auto_mask_generator.generate(image_rgb)
        masks.sort(key=lambda x: int(x.get("area", 0)), reverse=True)
        return masks


# ---------------------------------------------------------------------------
# Hierarchical Segmenter  — class-agnostic bootstrap + recursive drill-down
# ---------------------------------------------------------------------------

class HierarchicalSegmenter:

    def __init__(
        self,
        gdino: GroundingDINODetector,
        sam: SAMSegmenter,
        max_depth: int = 5,
        box_threshold: float = 0.28,
        text_threshold: float = 0.23,
        min_region_px: int = 800,
        iou_nms_threshold: float = 0.75,
        root_discovery_mode: str = "class_agnostic",
        bootstrap_max_regions: int = 96,
        per_region_max_proposals: int = 32,
        auto_mask_min_area: int = 1000,
        auto_mask_iou_thresh: float = 0.86,
        auto_mask_stability_thresh: float = 0.92,
        auto_mask_points_per_side: int = 16,
        auto_mask_points_per_batch: int = 32,
        auto_mask_crop_n_layers: int = 0,
        auto_mask_crop_n_points_downscale_factor: int = 2,
        auto_mask_max_side: int = 1536,
        use_prompt_hierarchy: bool = True,
    ):
        self.gdino = gdino
        self.sam = sam
        self.max_depth = max_depth
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_region_px = min_region_px
        self.iou_nms_threshold = iou_nms_threshold
        self.root_discovery_mode = root_discovery_mode
        self.bootstrap_max_regions = bootstrap_max_regions
        self.per_region_max_proposals = per_region_max_proposals
        self.auto_mask_min_area = auto_mask_min_area
        self.auto_mask_iou_thresh = auto_mask_iou_thresh
        self.auto_mask_stability_thresh = auto_mask_stability_thresh
        self.auto_mask_points_per_side = auto_mask_points_per_side
        self.auto_mask_points_per_batch = auto_mask_points_per_batch
        self.auto_mask_crop_n_layers = auto_mask_crop_n_layers
        self.auto_mask_crop_n_points_downscale_factor = auto_mask_crop_n_points_downscale_factor
        self.auto_mask_max_side = auto_mask_max_side
        self.use_prompt_hierarchy = use_prompt_hierarchy
        self._node_counter = 0

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _nms(
        self, detections: List[Tuple[str, float, List[int]]]
    ) -> List[Tuple[str, float, List[int]]]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        kept: List[Tuple[str, float, List[int]]] = []
        for det in detections:
            _, _, box = det
            if all(self._iou(box, kbox) <= self.iou_nms_threshold for _, _, kbox in kept):
                kept.append(det)
        return kept

    def _next_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter:05d}"

    def _detect_in_region(
        self,
        image_rgb: np.ndarray,
        prompts: List[str],
        region_bbox: Optional[List[int]],
    ) -> List[Tuple[str, float, List[int]]]:
        if not prompts:
            return []

        if region_bbox is not None:
            rx1, ry1, rx2, ry2 = region_bbox
            if (rx2 - rx1) * (ry2 - ry1) < self.min_region_px:
                return []
            crop = image_rgb[ry1:ry2, rx1:rx2]
        else:
            rx1 = ry1 = 0
            crop = image_rgb

        raw = self.gdino.detect(
            Image.fromarray(crop),
            prompts,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        abs_dets = [
            (label, score, [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1])
            for label, score, (x1, y1, x2, y2) in raw
        ]
        return self._nms(abs_dets)

    def _sam_segment(
        self,
        image_rgb: np.ndarray,
        box: List[int],
        parent_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        H, W = image_rgb.shape[:2]
        x1, y1, x2, y2 = box
        try:
            mask = self.sam.predict_box(box)
        except Exception as e:
            print(f"    [SAM] warning – falling back to rect mask: {e}")
            mask = np.zeros((H, W), dtype=bool)
            mask[y1:y2, x1:x2] = True
        if parent_mask is not None:
            mask = mask & parent_mask
        return mask

    @staticmethod
    def _tight_bbox(mask: np.ndarray, fallback: List[int]) -> List[int]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return fallback
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _propose_in_region(
        self,
        image_rgb: np.ndarray,
        region_bbox: Optional[List[int]],
        parent_mask: Optional[np.ndarray],
        limit: int,
    ) -> List[Tuple[str, float, List[int]]]:
        if region_bbox is not None:
            rx1, ry1, rx2, ry2 = region_bbox
            if (rx2 - rx1) * (ry2 - ry1) < self.min_region_px:
                return []
            crop = image_rgb[ry1:ry2, rx1:rx2]
        else:
            rx1 = ry1 = 0
            rx2, ry2 = image_rgb.shape[1], image_rgb.shape[0]
            crop = image_rgb

        # Downscale large regions for class-agnostic proposal generation to avoid OOM.
        scaled_crop = crop
        sx = sy = 1.0
        crop_h, crop_w = crop.shape[:2]
        if parent_mask is None and self.auto_mask_max_side > 0:
            long_side = max(crop_h, crop_w)
            if long_side > self.auto_mask_max_side:
                scale = self.auto_mask_max_side / float(long_side)
                new_w = max(1, int(round(crop_w * scale)))
                new_h = max(1, int(round(crop_h * scale)))
                scaled_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                sx = crop_w / float(new_w)
                sy = crop_h / float(new_h)

        try:
            masks = self.sam.generate_masks(
                image_rgb=scaled_crop,
                min_mask_area=max(int(self.auto_mask_min_area / (sx * sy)), max(64, self.min_region_px // 8)),
                pred_iou_thresh=self.auto_mask_iou_thresh,
                stability_score_thresh=self.auto_mask_stability_thresh,
                points_per_side=self.auto_mask_points_per_side,
                points_per_batch=self.auto_mask_points_per_batch,
                crop_n_layers=self.auto_mask_crop_n_layers,
                crop_n_points_downscale_factor=self.auto_mask_crop_n_points_downscale_factor,
            )
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retry with aggressively reduced settings.
            masks = self.sam.generate_masks(
                image_rgb=scaled_crop,
                min_mask_area=max(int(self.auto_mask_min_area / (sx * sy)), max(64, self.min_region_px // 8)),
                pred_iou_thresh=max(0.75, self.auto_mask_iou_thresh - 0.06),
                stability_score_thresh=max(0.80, self.auto_mask_stability_thresh - 0.08),
                points_per_side=max(8, self.auto_mask_points_per_side // 2),
                points_per_batch=max(8, self.auto_mask_points_per_batch // 2),
                crop_n_layers=0,
                crop_n_points_downscale_factor=max(2, self.auto_mask_crop_n_points_downscale_factor),
            )

        proposals: List[Tuple[str, float, List[int]]] = []
        for m in masks:
            x, y, w, h = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
            if sx != 1.0 or sy != 1.0:
                x = int(round(x * sx))
                y = int(round(y * sy))
                w = int(round(w * sx))
                h = int(round(h * sy))
            if w * h < self.min_region_px:
                continue

            x1, y1, x2, y2 = x + rx1, y + ry1, x + w + rx1, y + h + ry1
            x2 = min(x2, image_rgb.shape[1])
            y2 = min(y2, image_rgb.shape[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            if x2 <= x1 or y2 <= y1:
                continue

            if parent_mask is not None:
                seg = m.get("segmentation", None)
                if seg is None:
                    continue
                pm = parent_mask[ry1:ry2, rx1:rx2]
                overlap = int(np.logical_and(seg, pm).sum())
                seg_area = int(np.count_nonzero(seg))
                if seg_area == 0:
                    continue
                if overlap < max(self.min_region_px // 2, 64):
                    continue
                if overlap / max(seg_area, 1) < 0.35:
                    continue

            if region_bbox is not None:
                parent_area = (region_bbox[2] - region_bbox[0]) * (region_bbox[3] - region_bbox[1])
                child_area = (x2 - x1) * (y2 - y1)
                if child_area >= 0.97 * parent_area:
                    continue
                if self._iou([x1, y1, x2, y2], region_bbox) >= 0.97:
                    continue

            score = float(m.get("predicted_iou", 0.0)) * float(m.get("stability_score", 0.0))
            if score <= 0:
                score = float(m.get("stability_score", 0.0))
            proposals.append(("object", score, [x1, y1, x2, y2]))

        proposals = self._nms(proposals)
        if limit > 0:
            proposals = proposals[:limit]
        return proposals

    def _discover_children(
        self,
        image_rgb: np.ndarray,
        label: str,
        parent_box: List[int],
        parent_mask: np.ndarray,
    ) -> List[Tuple[str, float, List[int]]]:
        sub_bbox = self._tight_bbox(parent_mask, parent_box)

        # Prefer semantic hierarchy prompts if available.
        if self.use_prompt_hierarchy:
            child_prompts = get_child_prompts(label)
            if child_prompts:
                child_dets = self._detect_in_region(image_rgb, child_prompts, sub_bbox)
                if child_dets:
                    return child_dets

        # Fallback: class-agnostic proposals (no predefined classes).
        raw_children = self._propose_in_region(
            image_rgb=image_rgb,
            region_bbox=sub_bbox,
            parent_mask=parent_mask,
            limit=self.per_region_max_proposals,
        )
        children: List[Tuple[str, float, List[int]]] = []
        for idx, (_, score, box) in enumerate(raw_children, 1):
            children.append((f"{label}_part_{idx}", score, box))
        return children

    def _build_node(
        self,
        image_rgb: np.ndarray,
        label: str,
        score: float,
        box: List[int],
        depth: int,
        parent_mask: Optional[np.ndarray],
    ) -> Optional[Detection]:
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < self.min_region_px:
            return None

        mask = self._sam_segment(image_rgb, box, parent_mask)
        node = Detection(
            label=label,
            score=score,
            bbox_xyxy=box,
            mask=mask,
            depth=depth,
            node_id=self._next_id(),
        )

        if depth >= self.max_depth:
            return node

        children_dets = self._discover_children(
            image_rgb=image_rgb,
            label=label,
            parent_box=box,
            parent_mask=mask,
        )

        if not children_dets:
            return node

        children: List[Detection] = []
        for c_label, c_score, c_box in children_dets:
            child_node = self._build_node(
                image_rgb=image_rgb,
                label=c_label,
                score=c_score,
                box=c_box,
                depth=depth + 1,
                parent_mask=mask,
            )
            if child_node is not None:
                children.append(child_node)
        node.children = children
        return node

    def run(self, image_rgb: np.ndarray) -> List[Detection]:
        self._node_counter = 0

        # Set SAM image once for box-prompted segmentation throughout recursion.
        self.sam.set_image(image_rgb)

        if self.root_discovery_mode == "class_agnostic":
            print("  [Phase 1] Root discovery: class-agnostic SAM proposals")
            try:
                root_props = self._propose_in_region(
                    image_rgb=image_rgb,
                    region_bbox=None,
                    parent_mask=None,
                    limit=self.bootstrap_max_regions,
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("  [WARN] OOM in class-agnostic root discovery. Falling back to prompted root sweep.")
                root_prompts = PROMPT_HIERARCHY["__root__"]
                root_props = self._detect_in_region(image_rgb, root_prompts, None)
            root_dets = [
                (f"object_{idx:03d}", score, box)
                for idx, (_, score, box) in enumerate(root_props, 1)
            ]
        else:
            root_prompts = PROMPT_HIERARCHY["__root__"]
            print(f"  [Phase 1] Root sweep: {len(root_prompts)} categories → ", end="", flush=True)
            root_dets = self._detect_in_region(image_rgb, root_prompts, None)
            print(f"{len(root_dets)} object(s) found")

        if not root_dets:
            return []

        print(f"  [Phase 2] Drill-down from {len(root_dets)} discovered root object(s)")

        nodes: List[Detection] = []
        for idx, (label, score, box) in enumerate(root_dets, 1):
            print(f"    [{idx}/{len(root_dets)}] '{label}' (score={score:.3f})")
            node = self._build_node(
                image_rgb=image_rgb,
                label=label,
                score=score,
                box=box,
                depth=0,
                parent_mask=None,
            )
            if node is not None:
                nodes.append(node)

        return nodes


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def depth_color(depth: int) -> Tuple[int, int, int]:
    return PALETTE[depth % len(PALETTE)]


def _draw_nodes(
    canvas: np.ndarray,
    nodes: List[Detection],
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = canvas.copy()
    for node in nodes:
        if node.mask is not None:
            color = depth_color(node.depth)
            overlay[node.mask] = color
        canvas = _draw_nodes(canvas, node.children, alpha=alpha)

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    for node in nodes:
        x1, y1, x2, y2 = node.bbox_xyxy
        color = depth_color(node.depth)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_str = f"[{node.depth}] {node.label} {node.score:.2f}"
        cv2.putText(canvas, label_str, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        canvas = _draw_nodes(canvas, node.children, alpha=alpha)

    return canvas


def visualise(image_rgb: np.ndarray, nodes: List[Detection]) -> np.ndarray:
    canvas = image_rgb.copy()
    return _draw_nodes(canvas, nodes, alpha=0.35)


def count_nodes(nodes: List[Detection]) -> int:
    return sum(1 + count_nodes(n.children) for n in nodes)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_results(
    image_rgb: np.ndarray,
    nodes: List[Detection],
    out_dir: Path,
    image_stem: str,
    save_individual_masks: bool = True,
    save_crops: bool = True,
):
    """Save visualisation, JSON tree, and optionally per-node masks/crops."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Visualisation overlay
    vis = visualise(image_rgb, nodes)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f"{image_stem}_vis.jpg"), vis_bgr)

    # JSON tree
    tree = [n.to_dict() for n in nodes]
    with open(out_dir / f"{image_stem}_tree.json", "w") as f:
        json.dump(tree, f, indent=2)

    if save_individual_masks or save_crops:
        masks_dir = out_dir / image_stem / "masks"
        crops_dir = out_dir / image_stem / "crops"
        masks_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        def _save_node(node: Detection):
            safe_label = node.label.replace(" ", "_").replace("/", "-")
            stem = f"d{node.depth}_{safe_label}_{node.node_id}"

            if save_individual_masks and node.mask is not None:
                mask_img = (node.mask.astype(np.uint8)) * 255
                cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), mask_img)

            if save_crops:
                x1, y1, x2, y2 = node.bbox_xyxy
                crop_bgr = cv2.cvtColor(image_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crops_dir / f"{stem}_crop.jpg"), crop_bgr)

                if node.mask is not None:
                    masked = image_rgb.copy()
                    masked[~node.mask] = 0
                    masked_crop = masked[y1:y2, x1:x2]
                    masked_crop_bgr = cv2.cvtColor(masked_crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(crops_dir / f"{stem}_masked.jpg"), masked_crop_bgr)

            for child in node.children:
                _save_node(child)

        for n in nodes:
            _save_node(n)


def print_tree(nodes: List[Detection], indent: int = 0):
    for node in nodes:
        pad = "  " * indent
        print(f"{pad}{'└─' if indent else '┌─'} "
              f"[depth={node.depth}] {node.label!r:30s} score={node.score:.3f}  "
              f"box={node.bbox_xyxy}  children={len(node.children)}")
        print_tree(node.children, indent + 1)


# ---------------------------------------------------------------------------
# CLI Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # I/O
    images_dir: Path
    """Directory containing input images (jpg/png/bmp/tiff)."""

    output_dir: Path = Path("outputs/hierarchical_segments")
    """Root output directory. Per-image subdirectories are created automatically."""

    # Model selection
    gdino_model: str = "IDEA-Research/grounding-dino-base"
    """HuggingFace model ID for Grounding DINO.
    Options: IDEA-Research/grounding-dino-tiny | IDEA-Research/grounding-dino-base"""

    sam_model_type: str = "vit_h"
    """SAM v1 model variant: vit_h (best), vit_l, vit_b (fastest)."""

    sam_checkpoint: Optional[str] = None
    """Path to SAM checkpoint (.pth). Auto-downloaded to ~/.cache/sam if not set."""

    sam_cache_dir: str = "~/.cache/sam"
    """Directory to store auto-downloaded SAM checkpoints."""

    # Detection thresholds
    box_threshold: float = 0.28
    """Grounding DINO bounding-box confidence threshold (0–1)."""

    text_threshold: float = 0.23
    """Grounding DINO text-matching threshold (0–1)."""

    # Segmentation depth
    max_depth: int = 3
    """Maximum recursion depth (0 = scene, 1 = object parts, 2 = sub-parts, …).
    Set to a higher value (e.g. 5) to go deeper — runtime grows exponentially."""

    min_region_px: int = 800
    """Minimum bounding-box area (pixels²) for a region to be recursed into."""

    iou_nms_threshold: float = 0.75
    """IoU threshold for non-maximum suppression within each detection level."""

    # Output options
    save_individual_masks: bool = True
    """Save per-node binary mask PNGs."""

    save_crops: bool = True
    """Save per-node crop and masked-crop JPEGs."""

    # Misc
    device: str = "cuda"
    """Torch device: 'cuda' or 'cpu'. CUDA strongly recommended."""

    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    """File extensions to process."""

    max_images: Optional[int] = None
    """Limit processing to the first N images (useful for quick tests)."""

    prompt_chunk_size: int = 20
    """Max number of text categories sent to Grounding DINO in one call.
    Grounding DINO has a ~256-token text limit; large prompt lists are split
    into chunks of this size and merged. Lower = safer / faster per call;
    higher = fewer forward passes. 15–25 is a good range."""

    root_discovery_mode: str = "class_agnostic"
    """How to discover top-level objects.
    - class_agnostic: no predefined classes in phase-1 (SAM automatic masks)
    - prompted: use PROMPT_HIERARCHY['__root__'] with Grounding DINO"""

    bootstrap_max_regions: int = 96
    """Maximum number of root proposals in class_agnostic mode."""

    per_region_max_proposals: int = 32
    """Maximum class-agnostic child proposals per node during recursion."""

    auto_mask_min_area: int = 1000
    """SAM automatic-mask minimum region area in pixels."""

    auto_mask_iou_thresh: float = 0.86
    """SAM automatic-mask predicted IoU threshold."""

    auto_mask_stability_thresh: float = 0.92
    """SAM automatic-mask stability score threshold."""

    auto_mask_points_per_side: int = 16
    """SAM automatic-mask points per side. Lower values reduce VRAM use."""

    auto_mask_points_per_batch: int = 32
    """SAM automatic-mask batch size for point prompts. Lower reduces VRAM."""

    auto_mask_crop_n_layers: int = 0
    """SAM automatic-mask crop layers. Keep 0 for lower VRAM and faster runtime."""

    auto_mask_crop_n_points_downscale_factor: int = 2
    """Downscale factor for points in crop layers (when crop_n_layers > 0)."""

    auto_mask_max_side: int = 1536
    """Max image side for class-agnostic SAM proposal generation; larger images are resized."""

    use_prompt_hierarchy: bool = True
    """If True, try semantic child prompts first, then class-agnostic fallback."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_images(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(sorted(directory.rglob(f"*{ext}")))
        paths.extend(sorted(directory.rglob(f"*{ext.upper()}")))
    return sorted(set(paths))


def main(cfg: PipelineConfig) -> None:

    print("=" * 70)
    print("  Hierarchical Scene Segmentation  (Grounding DINO + SAM v1)")
    print("=" * 70)
    print(f"  images_dir      : {cfg.images_dir}")
    print(f"  output_dir      : {cfg.output_dir}")
    print(f"  gdino_model     : {cfg.gdino_model}")
    print(f"  sam_model_type  : {cfg.sam_model_type}")
    print(f"  max_depth       : {cfg.max_depth}")
    print(f"  box_threshold   : {cfg.box_threshold}")
    print(f"  text_threshold  : {cfg.text_threshold}")
    print(f"  root_mode       : {cfg.root_discovery_mode}")
    print(f"  device          : {cfg.device}")
    print()

    # ---- Collect images ---------------------------------------------------
    images = find_images(cfg.images_dir, cfg.image_extensions)
    if not images:
        print(f"[ERROR] No images found in {cfg.images_dir}")
        sys.exit(1)

    if cfg.max_images is not None:
        images = images[: cfg.max_images]

    print(f"Found {len(images)} image(s) to process.\n")

    # ---- Load models ------------------------------------------------------
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    gdino = GroundingDINODetector(
        model_id=cfg.gdino_model,
        device=device,
        prompt_chunk_size=cfg.prompt_chunk_size,
    )
    sam = SAMSegmenter(
        checkpoint=cfg.sam_checkpoint,
        model_type=cfg.sam_model_type,
        device=device,
        cache_dir=cfg.sam_cache_dir,
    )

    segmenter = HierarchicalSegmenter(
        gdino=gdino,
        sam=sam,
        max_depth=cfg.max_depth,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
        min_region_px=cfg.min_region_px,
        iou_nms_threshold=cfg.iou_nms_threshold,
        root_discovery_mode=cfg.root_discovery_mode,
        bootstrap_max_regions=cfg.bootstrap_max_regions,
        per_region_max_proposals=cfg.per_region_max_proposals,
        auto_mask_min_area=cfg.auto_mask_min_area,
        auto_mask_iou_thresh=cfg.auto_mask_iou_thresh,
        auto_mask_stability_thresh=cfg.auto_mask_stability_thresh,
        auto_mask_points_per_side=cfg.auto_mask_points_per_side,
        auto_mask_points_per_batch=cfg.auto_mask_points_per_batch,
        auto_mask_crop_n_layers=cfg.auto_mask_crop_n_layers,
        auto_mask_crop_n_points_downscale_factor=cfg.auto_mask_crop_n_points_downscale_factor,
        auto_mask_max_side=cfg.auto_mask_max_side,
        use_prompt_hierarchy=cfg.use_prompt_hierarchy,
    )

    # ---- Process each image ----------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Processing: {img_path.name}")
        t0 = time.time()

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        image_rgb = np.array(pil_img)
        H, W = image_rgb.shape[:2]
        print(f"  Size: {W}×{H}")

        # Run hierarchical segmentation
        try:
            nodes = segmenter.run(image_rgb)
        except Exception as e:
            print(f"  [ERROR] Segmentation failed: {e}")
            import traceback; traceback.print_exc()
            continue

        total_nodes = count_nodes(nodes)
        elapsed = time.time() - t0

        print(f"  Detected {len(nodes)} scene-level objects  |  "
              f"{total_nodes} total nodes across all depths  |  "
              f"{elapsed:.1f}s")

        # Print tree to console
        print_tree(nodes)

        # Save outputs
        out_dir = cfg.output_dir / img_path.stem
        save_results(
            image_rgb=image_rgb,
            nodes=nodes,
            out_dir=out_dir,
            image_stem=img_path.stem,
            save_individual_masks=cfg.save_individual_masks,
            save_crops=cfg.save_crops,
        )

        summary.append({
            "image": str(img_path),
            "width": W,
            "height": H,
            "scene_level_objects": len(nodes),
            "total_nodes": total_nodes,
            "elapsed_s": round(elapsed, 2),
        })

        print(f"  Saved to: {out_dir}")

    # ---- Write summary JSON ----------------------------------------------
    summary_path = cfg.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": {
            "gdino_model": cfg.gdino_model,
            "sam_model_type": cfg.sam_model_type,
            "max_depth": cfg.max_depth,
            "box_threshold": cfg.box_threshold,
            "text_threshold": cfg.text_threshold,
        }, "results": summary}, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done. Summary written to {summary_path}")
    total_objects = sum(r["total_nodes"] for r in summary)
    print(f"  Total objects/segments across all images: {total_objects}")
    print("=" * 70)


if __name__ == "__main__":
    cfg = tyro.cli(PipelineConfig)
    main(cfg)
