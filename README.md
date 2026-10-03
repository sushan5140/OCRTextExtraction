# OCRTextExtraction: Modern Spanish Handwriting OCR

An end-to-end Python pipeline capable of transcribing **modern handwritten Spanish text** (including cursive styles, accents `á, é, í, ó, ú`, and `ñ`) from raw images, exporting structured results to `.txt` and `.pdf`.

Powered by a fine-tuned Vision-Encoder-Decoder model hosted on Hugging Face:  
**Model Weights:** 
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ifesther/trocr-spanish-handwritten)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Key Features

- **Zero English Hallucination:** standard TrOCR decoders frequently hallucinate English phrases when reading Spanish. This model is fine-tuned to adhere to modern Spanish vocabulary, syntax, and sentence structure.
- **Pre-trained Weights Included:** uses [`ifesther/trocr-spanish-handwritten`](https://huggingface.co/ifesther/trocr-spanish-handwritten) hosted on Hugging Face. **No training required to use this tool.**
- **Document Segmentation:** employs morphological OpenCV filters to extract individual text lines from full notebook pages.
- **Direct Export:** converts raw scans into `.txt` and `.pdf` reports.

---

## Installation & Setup

1. **Clone the repository:**
   ```
   git clone [https://github.com/estherg12/OCRTextExtraction.git](https://github.com/estherg12/OCRTextExtraction.git)
   cd OCRTextExtraction
   ```
2. Create and activate a virtual environment:
   ```python -m venv .venv```
   - Windows: 
  ```.venv\Scripts\activate```
   - Linux/macOS:
  ```source .venv/bin/activate```
3. Install dependencies:
  ```pip install -r requirements.txt```
4. **Run Transcription:** transcribe an image or an entire folder of handwriting images:
   - **Single image:**
     ```bash
     python main.py --image images/test3.png
     # Tune TrOCR line batching for your available memory (default: 4)
     python main.py --image images/test3.png --batch-size 8
     ```
   - **Batch process an entire directory:**
     ```bash
     python main.py --folder images/ --batch-size 4
     # or using batch_process.py:
     python batch_process.py images/ --output output/ --batch-size 4
     ```

   `--batch-size` controls how many cropped text lines are sent through TrOCR in one forward pass. Use `--batch-size 1` for the previous single-line behavior or a larger value when memory allows.

---

## Contributing
Contributions are highly encouraged! Whether it's improving the segmentation algorithm, optimizing inference speed, or adding new features, please see our [CONTRIBUTING.md](https://github.com/estherg12/OCRTextExtraction/blob/main/CONTRIBUTING.md) for guidelines on how to get involved.

---

## Model Checkpoint
The core engine is based on a Vision-Encoder-Decoder fine-tuned on synthetic Spanish notebook text generated across 41 handwriting fonts.
- **Hugging Face Hub**: [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ifesther/trocr-spanish-handwritten)
- **Base Checkpoint**: ```microsoft/trocr-base-handwritten```
You can load the model directly in Python with Transformers:
```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained(
    "ifesther/trocr-spanish-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "ifesther/trocr-spanish-handwritten"
)
```

---

## Repository Architecture
- ```main.py```: entry point for full-page processing and document export (.txt / .pdf). From here you can change the desired photo to analyze. 
- ```generate_synthetic_dataset.py```: synthetic generator used to create diverse handwriting line crops, ending in a large dataset with 30 thousand crops to train the model.
- ```train_trocr.py```: fine-tuning script (only needed if retraining from scratch, I recommend running it from Google Collab).
- ```upload_model.py```: was used to upload the trained model to Hugging Face.
- ```local_app.py```: optional interactive visual UI.
- ```fonts/``` & ```training_pdf/```: input source materials.
- ```images/```: samples.
- ```count_lines.py``` & ```check_lines.py```: calculate the heights, top and bottom pixels from each line detected in a photo. Then crops each detected text line from the original image and saves it as a PNG. They will all be used by ```main.py``` to individually predict its text and then put it toghether.

---

## License
This project is licensed under the MIT License.
