import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remplace les caractères non-ASCII par un espace
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def clean_detection_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df['func'] = df['func'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Detection dataset cleaned and saved to {output_path}")

def clean_patch_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df['code_before'] = df['code_before'].apply(clean_text)
    df['code_after'] = df['code_after'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Patch dataset cleaned and saved to {output_path}")

if __name__ == "__main__":
    base = Path('.')
    clean_detection_dataset(base / 'dataset_detection.csv', base / 'dataset_detection_clean.csv')
    clean_patch_dataset(base / 'dataset_patch.csv', base / 'dataset_patch_clean.csv') 