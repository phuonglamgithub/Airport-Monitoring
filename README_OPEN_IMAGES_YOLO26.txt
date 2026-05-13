YOLO26 + Open Images V7 support
================================

This demo is an inference application, so it does not need the full open-images-v7.yaml file at runtime.
The YAML file is mainly used when training a YOLO model.

What was changed:
1. The detector no longer assumes COCO class IDs such as Person=0 or Suitcase=28.
2. Class IDs are resolved dynamically from the loaded checkpoint's model.names.
3. This makes the app compatible with YOLO26 / Open Images V7 checkpoints whose class IDs are different from COCO.
4. UI labels are mapped as follows:
   - Person: Person, Man, Woman, Boy, Girl, Human, Pedestrian, People
   - Luggage: Suitcase, Backpack, Handbag, Briefcase, Bag, Purse, Luggage and Bags
   - Pet: Dog, Cat, Rabbit, Hamster, Guinea Pig, Bird
   - Hurry Up/Busy: derived from Person movement speed, not a standalone dataset class

Training reference:
If you train with Ultralytics, use the official dataset config:

    yolo detect train model=yolo26n.pt data=open-images-v7.yaml imgsz=640 epochs=50

For this Flask demo, place your trained .pt file inside the models/ folder, then load it from the UI.
