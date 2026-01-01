#!/usr/bin/env python3
"""
Validate voice catalog against official Hugging Face voices.json
This script downloads the official voices.json and updates our ALL_VOICES catalog
"""

import requests
import json
from pathlib import Path

# Download official voices.json
print("Downloading official voices.json from Hugging Face...")
try:
    response = requests.get(
        "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json",
        timeout=30
    )
    response.raise_for_status()
    official_voices = response.json()
    print(f"✓ Downloaded voices.json with {len(official_voices)} voices")
except Exception as e:
    print(f"✗ Error downloading voices.json: {e}")
    exit(1)

# Organize voices by language
voices_by_lang = {}
for voice_name, voice_info in official_voices.items():
    # Extract language code from voice name (first part before dash)
    lang_code = voice_name.split('-')[0]  # e.g., en_US, es_ES
    
    if lang_code not in voices_by_lang:
        voices_by_lang[lang_code] = []
    voices_by_lang[lang_code].append(voice_name)

# Sort voices within each language
for lang_code in voices_by_lang:
    voices_by_lang[lang_code].sort()

print(f"\n✓ Organized into {len(voices_by_lang)} languages")
print("\nLanguage breakdown:")
for lang in sorted(voices_by_lang.keys()):
    count = len(voices_by_lang[lang])
    print(f"  {lang}: {count} voices")

total_voices = sum(len(v) for v in voices_by_lang.values())
print(f"\nTotal voices: {total_voices}")

# Now let's generate the Python code for ALL_VOICES
print("\n" + "="*60)
print("Generated ALL_VOICES dictionary:")
print("="*60 + "\n")

print("ALL_VOICES = {")
for lang in sorted(voices_by_lang.keys()):
    voices = voices_by_lang[lang]
    print(f'    "{lang}": [')
    for voice in voices:
        print(f'        "{voice}",')
    print("    ],")
print("}")

# Save to file for easy copying
output_file = Path(__file__).parent / "all_voices_updated.py"
with open(output_file, 'w') as f:
    f.write("ALL_VOICES = {\n")
    for lang in sorted(voices_by_lang.keys()):
        voices = voices_by_lang[lang]
        f.write(f'    "{lang}": [\n')
        for voice in voices:
            f.write(f'        "{voice}",\n')
        f.write("    ],\n")
    f.write("}\n")

print(f"\n✓ Saved to: {output_file}")
print("\nYou can now update voice_assistant.py with the generated ALL_VOICES dictionary")
