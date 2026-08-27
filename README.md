# CBIS-DDSM Breast Cancer CNN App

PyTorch CNN project for binary classification of CBIS-DDSM mammogram JPG images.

## Important
This is an educational/research project, not a clinical diagnostic system.

## What the project implements
- PyTorch CNN
- Conv2d -> BatchNorm2d -> ReLU -> MaxPool
- dropout
- data augmentation
- weight decay
- Adam optimizer
- CrossEntropyLoss with class weighting
- patient-grouped validation split
- early stopping
- best validation checkpoint
- held-out CBIS-DDSM test evaluation
- sensitivity, specificity, ROC-AUC and classification report
- Streamlit prediction app

The design follows the supplied lecture note's CNN, preprocessing, validation, regularization and evaluation workflow.

## Dataset
The supplied CSVs were inspected and combined:
- mass_case(with_jpg_img).csv
- calc_case(with_jpg_img).csv
- metadata(with_jpg_img).csv

The model uses `jpg_fullMammo_img_path` because the project target is mammogram-level classification.

Binary labels:
- BENIGN + BENIGN_WITHOUT_CALLBACK -> 0 (benign)
- MALIGNANT -> 1 (malignant)

To avoid ambiguous image-level targets, full-mammogram paths that occur with contradictory pathology labels are excluded from the clean metadata index.

The CBIS-DDSM paths already distinguish Training and Test images. The official Test paths are retained as the final test set. The Training portion is split into train/validation at patient level.

## 1. Put the dataset on disk

The CSV paths look like:

jpg_img/...

So choose the folder that contains `jpg_img`, for example:

C:/datasets/CBIS-DDSM/

Then verify:

C:/datasets/CBIS-DDSM/jpg_img/...

## 2. Prepare the index

```bash
python prepare_dataset.py --image-root "C:/datasets/CBIS-DDSM"
```

This creates `data_index.csv`.

## 3. Train

```bash
python train.py
```

For a GPU:

```bash
python train.py --batch-size 32 --epochs 30
```

If GPU memory is limited:

```bash
python train.py --batch-size 8
```

## 4. Run the app

```bash
streamlit run app.py
```

## 5. Results

Training history:
`results/training_history.csv`

Final test metrics:
`results/test_metrics.json`

Best model:
`models/best_breast_cancer_cnn.pth`

## Note on mammogram resolution
The supplied lecture uses small resized images for demonstrating CNN workflows. This project follows that teaching pattern with 128x128 grayscale inputs. For a serious research-grade mammography model, higher-resolution/tiled or lesion-focused processing would normally be investigated separately.
