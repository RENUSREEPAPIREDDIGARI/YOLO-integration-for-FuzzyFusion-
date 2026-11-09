# Classification dataset

The fuzzy-fusion project uses class-folder labels, which are suitable for
YOLO **image classification**. They do not contain bounding boxes and cannot
train an object detector.

The training script reads the source dataset without changing it, then copies
images into a deterministic train/validation/test split under the ignored
`data/colorectal-cls/` directory:

```text
data/colorectal-cls/
  train/
    class-a/
    class-b/
  val/
    class-a/
    class-b/
  test/
    class-a/
    class-b/
```

Provide the source folder with `--source`. It must contain one folder per
class, with supported image files inside each class folder. The default source
in `config.yaml` points to the sibling `CI Project/kvasir-dataset` folder; use
`--source` if yours is elsewhere.

The split is reproducible with the configured seed and ratios. Re-running
training regenerates only the marked `data/colorectal-cls/` prepared copy; it
does not modify the source images.
