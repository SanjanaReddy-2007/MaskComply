# MaskComply

**MaskComply** is a computer-vision-based safety compliance system that detects whether people are wearing masks correctly and converts detection results into a **compliance/risk score**.

The project combines **YOLO-based person detection, mask classification, video processing, and compliance scoring** to identify unsafe mask-wearing behavior and generate compliance information.

---

## Features

- Detects people in video frames.
- Classifies mask usage into:
  - `with_mask`
  - `without_mask`
  - `incorrect_mask`
- Uses a fine-tuned image classification model for mask classification.
- Processes videos frame-by-frame.
- Calculates compliance scores based on observed mask behavior.
- Generates compliance logs for analysis.
- Provides a Streamlit-based dashboard for visualization.
- Supports synthetic video testing for validating the compliance-scoring layer.

---

## Project Structure

```text
MaskComply/
│
├── data/
│   └── shiekhburhan_raw/
│       └── FMD_DATASET/
│           ├── incorrect_mask/
│           ├── with_mask/
│           └── without_mask/
│
├── docs/
│   └── DAY2_README.md
│
├── models/
│   ├── classes.json
│   └── mask_classifier.pt
│
├── outputs/
│   ├── pic1_annotated.jpg
│   ├── pic2_annotated.jpg
│   ├── synthetic_compliance_log.csv
│   └── synthetic_scored.mp4
│
├── src/
│   ├── dashboard.py
│   ├── detect.py
│   ├── nms.py
│   ├── pipeline_video.py
│   ├── scorer.py
│   ├── tracker.py
│   ├── train_classifier.py
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md

Dataset

The project uses the Face Mask Dataset by shiekhburhan from Kaggle.

The dataset contains three categories:

incorrect_mask
with_mask
without_mask

The dataset is not included in this repository because of its large size.

Download Dataset

Install the Kaggle CLI if required and authenticate using your Kaggle API token.

Then run:

kaggle datasets download shiekhburhan/face-mask-dataset

Create the dataset directory:

New-Item -ItemType Directory -Force data\shiekhburhan_raw

Extract the downloaded ZIP:

Expand-Archive -Path .\face-mask-dataset.zip -DestinationPath .\data\shiekhburhan_raw -Force

The expected structure is:

data/
└── shiekhburhan_raw/
    └── FMD_DATASET/
        ├── incorrect_mask/
        ├── with_mask/
        └── without_mask/
Installation

Clone the repository:

git clone https://github.com/SanjanaReddy-2007/MaskComply.git
cd MaskComply

Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Train the Mask Classifier

The classifier can be trained using the prepared dataset:

python -m src.train_classifier --data_dir .\data\shiekhburhan_raw\FMD_DATASET

The trained model is saved as:

models/mask_classifier.pt

Class information is stored in:

models/classes.json
Run Video Processing

The video pipeline processes an input video and applies detection, tracking, mask classification, and compliance scoring.

Example:

python src/pipeline_video.py

Refer to the source script for the available configuration options.

Compliance Scoring

Mask classification alone does not represent overall compliance.

The scoring layer considers observed behavior over time and converts detections into a compliance/risk measure.

For example:

With Mask          → Compliant
Incorrect Mask     → Partial / Reduced Compliance
Without Mask       → Non-Compliant

The scoring layer can aggregate observations across frames or tracked individuals rather than treating every frame independently.

This makes the system more suitable for continuous video monitoring.

Streamlit Dashboard

The project includes a Streamlit dashboard:

streamlit run src\dashboard.py

The dashboard is intended to provide an interface for viewing video-processing and compliance results.

Outputs

Generated outputs may include:

Annotated Images
outputs/pic1_annotated.jpg
outputs/pic2_annotated.jpg
Compliance Log
outputs/synthetic_compliance_log.csv

The log contains compliance-related information generated during processing.

Processed Video
outputs/synthetic_scored.mp4

The processed video contains the scoring/annotation results.

Technologies Used
Python
PyTorch
Torchvision
OpenCV
Ultralytics YOLO
NumPy
Pandas
Streamlit
Kaggle Dataset
Future Improvements
Improve mask classification accuracy under difficult lighting and occlusion.
Improve tracking consistency across frames.
Add real-time camera support.
Improve individual-level compliance tracking.
Add configurable compliance thresholds.
Extend the scoring system for long-term monitoring and analytics.
Deploy the dashboard as a web application.
License

This project is intended for academic and research purposes.

The face-mask dataset used by the project is obtained from Kaggle and follows the license specified by its original dataset provider.


### One thing I'd change before pushing

Your repository currently has:

```text
outputs/
    synthetic_scored.mp4
    synthetic_compliance_log.csv
    ...

but your .gitignore is configured to ignore generated outputs. That's actually good—you don't need to commit generated videos/CSVs.