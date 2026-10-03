import os
from pathlib import Path
from count_lines import measure_text_lines, crop_and_save_lines

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def get_image_files(directory_path: str | Path) -> list[Path]:
    """
    Returns a sorted list of all supported image files (.jpg, .jpeg, .png)
    found inside directory_path (non-recursive).
    """
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    images = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name)


def cleanup_line_crops(output_dir: str | Path = "output", prefix: str | None = None) -> None:
    """Deletes temporary cropped line images while preserving final transcriptions."""
    out = Path(output_dir)
    if not out.exists():
        return
    for item in out.glob("*.png"):
        if "_line_" in item.name:
            if prefix is None or item.name.startswith(f"{prefix}_line_"):
                try:
                    item.unlink()
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")


def export_to_txt(lines: list[str], output_path: str | Path) -> None:
    """Export lines to a UTF-8 text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved transcript to: {output_path}")


def _sanitize_pdf_text(text: str) -> str:
    """Replaces Unicode punctuation not present in Latin-1 with safe substitutes."""
    replacements = {
        '—': '-', '–': '-', '―': '-',
        '“': '"', '”': '"', '«': '"', '»': '"',
        '‘': "'", '’': "'", '`': "'",
        '…': '...', '•': '*', '\xa0': ' '
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def export_to_pdf(
    lines: list[str],
    output_path: str | Path,
    line_image_paths: list[str | Path] | None = None,
) -> None:
    """
    Export transcript lines to PDF, optionally showing each source crop above its text.

    When ``line_image_paths`` is provided, it must contain one image for each
    transcript line so that every OCR result remains visually traceable to its
    corresponding source crop.
    """
    from fpdf import FPDF
    from PIL import Image

    if line_image_paths is not None and len(line_image_paths) != len(lines):
        raise ValueError("line_image_paths must contain one image per transcript line")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF(format='letter')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)

    for index, line in enumerate(lines):
        if line_image_paths is not None:
            image_path = Path(line_image_paths[index])
            if not image_path.exists():
                raise FileNotFoundError(f"Line crop not found: {image_path}")

            with Image.open(image_path) as image:
                pixel_width, pixel_height = image.size

            if pixel_width <= 0 or pixel_height <= 0:
                raise ValueError(f"Invalid line crop dimensions: {image_path}")

            max_width = min(pdf.epw, 170)
            max_height = 60
            scale = min(max_width / pixel_width, max_height / pixel_height)
            image_width = pixel_width * scale
            image_height = pixel_height * scale

            required_height = image_height + 16
            if pdf.get_y() + required_height > pdf.h - pdf.b_margin:
                pdf.add_page()

            image_x = (pdf.w - image_width) / 2
            image_y = pdf.get_y()
            pdf.image(
                str(image_path),
                x=image_x,
                y=image_y,
                w=image_width,
                h=image_height,
            )
            pdf.set_y(image_y + image_height + 2)

        safe_line = _sanitize_pdf_text(line)
        pdf.cell(0, 10, text=safe_line, align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    pdf.output(str(output_path))
    print(f"Saved transcript to: {output_path}")


def transcribe_lines(
    line_image_paths: list[str | Path],
    processor=None,
    model=None,
    device: str = "cpu",
    transcribe_fn=None,
    batch_size: int = 4,
) -> tuple[list[str], list[str]]:
    """Transcribe line crops while preserving their original order.

    Custom transcribe functions keep the existing one-image-at-a-time behavior.
    TrOCR processor/model pairs are run in configurable batches. Set batch_size
    to 1 to retain the previous single-line inference behavior.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if transcribe_fn is not None:
        raw_lines: list[str] = []
        nlp_lines: list[str] = []
        for i, line_img in enumerate(line_image_paths, start=1):
            raw, nlp = transcribe_fn(str(line_img), i)
            raw_lines.append(raw)
            nlp_lines.append(nlp)
        return raw_lines, nlp_lines

    if processor is None or model is None:
        raise ValueError("processor and model are required when transcribe_fn is not provided")

    import torch
    from PIL import Image
    from main import correct_spanish_text

    raw_lines: list[str] = []
    nlp_lines: list[str] = []

    for start in range(0, len(line_image_paths), batch_size):
        batch_paths = line_image_paths[start:start + batch_size]
        images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        pixel_values = processor(
            images=images,
            return_tensors="pt",
        ).pixel_values.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=32,
                num_beams=1,
            )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        raw_lines.extend(decoded)
        nlp_lines.extend(correct_spanish_text(text) for text in decoded)

    return raw_lines, nlp_lines


def process_image(
    image_path: str | Path,
    processor=None,
    model=None,
    output_dir: str | Path = "output",
    device: str = "cpu",
    transcribe_fn=None,
    batch_size: int = 4,
) -> dict[str, str]:
    """
    Processes a single image:
    1. Measures text lines using measure_text_lines()
    2. Crops and saves line segments into output_dir
    3. Transcribes each line segment
    4. Exports raw and NLP transcriptions as .txt and .pdf named after the input image
    5. Cleans up intermediate line crops after PDF generation
    """
    img_path = Path(image_path)
    if not img_path.exists():
        fallback = Path("images") / image_path
        if fallback.exists():
            img_path = fallback
        else:
            raise FileNotFoundError(f"Could not find image '{image_path}'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing Image: {img_path.name} ---")
    detected_lines = measure_text_lines(str(img_path))
    print(f"Detected {len(detected_lines)} line(s).")

    if not detected_lines:
        print(f"No lines detected in {img_path.name}.")
        return {}

    crop_and_save_lines(str(img_path), detected_lines, output_dir=str(out_dir))

    base_name = img_path.stem
    line_image_paths = [
        out_dir / f"{base_name}_line_{i}.png"
        for i in range(1, len(detected_lines) + 1)
    ]

    raw_lines, nlp_lines = transcribe_lines(
        line_image_paths,
        processor=processor,
        model=model,
        device=device,
        transcribe_fn=transcribe_fn,
        batch_size=batch_size,
    )

    # Primary output files named exactly after input image
    txt_main = out_dir / f"{base_name}.txt"
    pdf_main = out_dir / f"{base_name}.pdf"
    txt_raw = out_dir / f"{base_name}_transcription_raw.txt"
    pdf_raw = out_dir / f"{base_name}_transcription_raw.pdf"
    txt_nlp = out_dir / f"{base_name}_transcription_nlp.txt"
    pdf_nlp = out_dir / f"{base_name}_transcription_nlp.pdf"

    final_lines = nlp_lines if any(nlp_lines) else raw_lines

    try:
        export_to_txt(final_lines, txt_main)
        export_to_pdf(final_lines, pdf_main, line_image_paths=line_image_paths)
        export_to_txt(raw_lines, txt_raw)
        export_to_pdf(raw_lines, pdf_raw, line_image_paths=line_image_paths)
        export_to_txt(nlp_lines, txt_nlp)
        export_to_pdf(nlp_lines, pdf_nlp, line_image_paths=line_image_paths)
    finally:
        # Keep crops available until all PDFs have embedded them, then remove them.
        cleanup_line_crops(output_dir=out_dir, prefix=base_name)

    return {
        "txt": str(txt_main),
        "pdf": str(pdf_main),
        "txt_raw": str(txt_raw),
        "pdf_raw": str(pdf_raw),
        "txt_nlp": str(txt_nlp),
        "pdf_nlp": str(pdf_nlp),
    }


def batch_process(
    directory_path: str | Path,
    processor=None,
    model=None,
    output_dir: str | Path = "output",
    device: str = "cpu",
    transcribe_fn=None,
    batch_size: int = 4,
) -> list[dict[str, str]]:
    """
    Processes all supported images in a directory.
    """
    image_files = get_image_files(directory_path)
    print(f"Found {len(image_files)} image(s) to process in '{directory_path}'.")

    results = []
    for img_file in image_files:
        res = process_image(
            img_file,
            processor=processor,
            model=model,
            output_dir=output_dir,
            device=device,
            transcribe_fn=transcribe_fn,
            batch_size=batch_size,
        )
        if res:
            results.append(res)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process directory of images for OCR line extraction and transcription."
    )
    parser.add_argument("directory", nargs="?", default="images", help="Path to folder containing images (default: images)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--model-dir", "-m", default="./trocr_spanish_final", help="Path to local TrOCR model directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of cropped lines per TrOCR forward pass (default: 4)")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Local model directory '{args.model_dir}' missing.")

    import torch
    from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{args.model_dir}' on {device}...")
    image_processor = ViTImageProcessor.from_pretrained(args.model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    batch_process(
        args.directory,
        processor=processor,
        model=model,
        output_dir=args.output,
        device=device,
        batch_size=args.batch_size,
    )
