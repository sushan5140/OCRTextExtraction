import os
import shutil
from pathlib import Path
import re
import torch
from PIL import Image
from fpdf import FPDF
from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
from spellchecker import SpellChecker

# Target a single-phrase test image directly
IMAGE_FILE = "test3"
LOCAL_MODEL_DIR = "./trocr_spanish_final"


def correct_spanish_text(text: str) -> str:
    """
    Certifies words against the Spanish vocabulary and corrects typos/confusions while preserving:
    - acronyms
    - proper nouns
    - mixed-case names
    - alphanumeric identifiers
    - numbers
    - words containing hyphens

    Only ordinary lowercase words are sent to pyspellchecker.
    """

    spell = SpellChecker(language='es')

    words = text.split()
    corrected_words = []

    for word in words:

        # Separate punctuation from the actual token.
        match = re.match(
            r"^([^\wáéíóúüñÁÉÍÓÚÜÑ]*)([\wáéíóúüñÁÉÍÓÚÜÑ-]+)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)$",
            word,
        )

        if not match:
            corrected_words.append(word)
            continue

        prefix, clean_word, suffix = match.groups()

        # ---------------------------------------------------------
        # 1. Numbers / numeric identifiers
        # ---------------------------------------------------------
        if clean_word.isdigit():
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 2. Words containing digits
        # ---------------------------------------------------------
        if any(char.isdigit() for char in clean_word):
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 3. Acronyms / all-uppercase words
        # ---------------------------------------------------------
        if clean_word.isupper() and any(char.isalpha() for char in clean_word):
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 4. Mixed-case words
        # ---------------------------------------------------------
        if (
            any(char.isupper() for char in clean_word[1:])
            and any(char.islower() for char in clean_word)
        ):
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 5. Hyphenated words
        # ---------------------------------------------------------
        if "-" in clean_word:
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 6. Proper nouns
        # ---------------------------------------------------------
        if clean_word.istitle():
            corrected_words.append(word)
            continue

        # ---------------------------------------------------------
        # 7. Perform spell checking.
        # ---------------------------------------------------------
        normalized_word = clean_word.lower()

        unknowns = spell.unknown([normalized_word])

        if normalized_word in unknowns:
            correction = spell.correction(normalized_word)

            if correction:
                word = ( prefix + correction + suffix )

        corrected_words.append(word)

    return " ".join(corrected_words)


def test_single_phrase(
    image_path: str,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    i: int,
    device: str = "cpu"
) -> tuple[str, str]:
    # Locate image
    target_path = image_path
    if not Path(target_path).exists():
        target_path = f"images/{image_path}"
    if not Path(target_path).exists():
        raise FileNotFoundError(f"Could not find image '{image_path}' anywhere.")

    print(f"Predicting text for single-phrase image: '{target_path}'...")
    img = Image.open(target_path).convert("RGB")

    # Raw single-pass inference without any extra preprocessing
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=32,
            num_beams=1,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    corrected_text = correct_spanish_text(text)

    """
    print(f"\nLine {i}")
    print(f"Raw OCR Output    : {text.strip()}")
    print(f"NLP Certified Text: {corrected_text.strip()}")
    """

    return text, corrected_text


def delete_output_content():
    folder = "output"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def export_to_txt(lines: list[str], output_path: str):
    """Export lines to a UTF-8 text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved transcript to: {output_path}")


def export_to_pdf(lines: list[str], output_path: str):
    """Export lines to a PDF file handling Spanish characters and accents."""
    pdf = FPDF(format='letter')
    # pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)

    for line in lines:
        pdf.cell(200, 10, text=line,
                 ln=1, align='C')
        pdf.ln()

    pdf.output(output_path)
    print(f"Saved transcript to: {output_path}")


if __name__ == "__main__":
    import argparse
    from batch_process import batch_process, process_image

    parser = argparse.ArgumentParser(description="OCR handwritten line extraction and transcription.")
    parser.add_argument("path", nargs="?", default=None, help="Path to directory or image file to process")
    parser.add_argument("--folder", "-f", default=None, help="Directory containing images to batch process")
    parser.add_argument("--image", "-i", default=None, help="Path to single image file to process")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--model-dir", "-m", default=LOCAL_MODEL_DIR, help="Local model directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of cropped lines per TrOCR forward pass (default: 4)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Local model directory '{args.model_dir}' missing.")

    print(f"\nLoading local model from '{args.model_dir}' on {device}...")

    # Load processor, tokenizer, and model directly
    image_processor = ViTImageProcessor.from_pretrained(args.model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    folder_target = args.folder
    image_target = args.image

    if args.path:
        p = Path(args.path)
        if p.is_dir() or (not p.exists() and not p.suffix):
            folder_target = args.path
        else:
            image_target = args.path

    if folder_target:
        print(f"Starting batch processing on folder: {folder_target}")
        batch_process(
            folder_target,
            processor=processor,
            model=model,
            output_dir=args.output,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        target_image = image_target or f"images/{IMAGE_FILE}.png"
        print(f"Processing single image: {target_image}")
        process_image(
            target_image,
            processor=processor,
            model=model,
            output_dir=args.output,
            device=device,
            batch_size=args.batch_size,
        )
