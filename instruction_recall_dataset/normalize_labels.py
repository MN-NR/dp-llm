"""
normalize_labels.py — Normalize fine-grained PII labels to coarse labels

This script converts fine-grained labels (GIVENNAME, STREETADDRESS, etc.) 
to coarse-grained labels (NAME, ADDRESS, etc.) for consistent PII classification.

Supported label mappings:
  - Names: GIVENNAME, SURNAME, FIRSTNAME, LASTNAME, FULLNAME → NAME
  - Contact: EMAIL, PHONE, PHONENUMBER → EMAIL/PHONE
  - Address: BUILDINGNUM, STREETADDRESS, CITY, STATE, ZIPCODE → ADDRESS
  - Financial: CREDITCARDNUMBER → CREDIT_CARD, SSN → SSN
  - Other: URL, TIME, COMPANY

Usage:
    # Normalize JSONL file
    python normalize_labels.py --input data/audit/ai4privacy_records.jsonl \\
                               --output data/audit/ai4privacy_records_normalized.jsonl

    # Or use programmatically:
    from normalize_labels import normalize_sample, normalize_jsonl
    sample = normalize_sample({"target": "[GIVENNAME] Smith works at [COMPANY] ABC"})
    normalize_jsonl("input.jsonl", "output.jsonl")
"""

import json
import re
import sys
import os
import argparse
import logging

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    # Names
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "FIRSTNAME": "NAME",
    "LASTNAME": "NAME",
    "FULLNAME": "NAME",

    # Contact
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "TELEPHONENUM": "PHONE",

    # Address (ALL → ADDRESS)
    "BUILDINGNUM": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "STATE",
    "ZIPCODE": "ZIP_CODE",

    # Financial / sensitive
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "SSN": "SSN",

    # Other
    "URL": "URL",
    "TIME": "TIME",
    "COMPANY": "COMPANY"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_labels(text):
    """
    Convert fine-grained labels to coarse labels.
    
    Args:
        text (str): Text with fine-grained tags like [GIVENNAME], [STREETADDRESS]
        
    Returns:
        str: Text with normalized coarse tags like [NAME], [ADDRESS]
    """
    for fine, coarse in LABEL_MAP.items():
        text = re.sub(rf"\[{fine}\]", f"[{coarse}]", text)
    return text


def collapse_duplicates(text):
    """
    Collapse repeated tags like [NAME] [NAME] → [NAME].
    
    Removes consecutive duplicate tags and cleans up whitespace.
    
    Args:
        text (str): Text with potentially duplicate tags
        
    Returns:
        str: Text with collapsed duplicate tags
    """
    tags = [
        "NAME", "ADDRESS"
    ]
    
    for tag in tags:
        # Replace 2+ consecutive [TAG] with single [TAG]
        text = re.sub(rf"(\[{tag}\]\s*){{2,}}", f"[{tag}] ", text)
    
    return text.strip()


def normalize_sample(sample):
    """
    Apply normalization to one sample.
    
    Args:
        sample (dict): A dictionary with a "target" key containing text with PII labels
        
    Returns:
        dict: The sample with normalized labels
    """
    if "target" in sample:
        sample["target"] = normalize_labels(sample["target"])
        sample["target"] = collapse_duplicates(sample["target"])
    
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_jsonl(input_path, output_path, logger=None):
    """
    Normalize a JSONL dataset (recommended for large datasets).
    
    Processes line-by-line to minimize memory usage.
    
    Args:
        input_path (str): Path to input JSONL file
        output_path (str): Path to output JSONL file
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, desc=f"Normalizing {os.path.basename(input_path)}"):
            sample = json.loads(line)
            sample = normalize_sample(sample)
            f_out.write(json.dumps(sample) + "\n")
            count += 1
    
    logger.info(f"✅ Normalized {count} samples from {input_path}")
    logger.info(f"✅ Saved to: {output_path}")


def normalize_json(input_path, output_path, logger=None):
    """
    Normalize a JSON dataset.
    
    Loads entire dataset into memory - use normalize_jsonl for large files.
    
    Args:
        input_path (str): Path to input JSON file
        output_path (str): Path to output JSON file
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    normalized_data = []
    for sample in tqdm(data, desc=f"Normalizing {os.path.basename(input_path)}"):
        sample = normalize_sample(sample)
        normalized_data.append(sample)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2)
    
    logger.info(f"✅ Normalized {len(normalized_data)} samples from {input_path}")
    logger.info(f"✅ Saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir="data/audit"):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/normalize_labels.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize fine-grained PII labels to coarse labels"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input dataset path (JSONL or JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output dataset path (JSONL or JSON)"
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Determine format from file extension
    if args.input.endswith(".jsonl"):
        normalize_jsonl(args.input, args.output, logger)
    else:
        normalize_json(args.input, args.output, logger)


if __name__ == "__main__":
    main()
