# Image folder Torch classification

Small image-folder example using pure Torch and tiny PPM images.

```bash
uv lock
primejob dataset push data --disk pj-image-folder --subdir data
primejob run train.py --gpu CPU --disk pj-image-folder --data-mode stage --plain --yes
```

The script expects `PRIMEJOB_DATASET_PATH/data/image_folder/<class>/*.ppm`.
