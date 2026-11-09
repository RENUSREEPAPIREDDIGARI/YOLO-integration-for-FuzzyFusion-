# Dataset setup

Place a detection dataset in the following YOLO layout:

```text
data/colorectal/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

For each image, add a same-basename `.txt` file in the corresponding labels
folder. Each line is a bounding box in normalized YOLO format:

```text
class_id x_center y_center width height
```

Coordinates must be between 0 and 1. Update `colorectal.yaml` if your class
names or dataset directory differ. For example, `0 0.52 0.48 0.20 0.30`
marks one box for class `0`.

Do not commit private, identifiable, or unlicensed images or annotations.
