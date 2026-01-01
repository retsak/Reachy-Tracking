"""
Validate that all voices in the catalog have correct quality levels
by checking if they exist on Hugging Face
"""

import requests
from voice_assistant import ALL_VOICES

def check_voice_url(voice_name):
    """Check if a voice exists on Hugging Face"""
    parts = voice_name.split('-')
    if len(parts) < 3:
        return False, "Invalid format"
    
    lang_region = parts[0]
    quality = parts[-1]
    voice_name_part = '-'.join(parts[1:-1])
    
    if '_' not in lang_region:
        return False, "Invalid lang_region"
    
    lang_code = lang_region.split('_')[0]
    url_path = f"{lang_code}/{lang_region}/{voice_name_part}/{quality}"
    url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{url_path}/{voice_name}.onnx"
    
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        exists = response.status_code in [200, 302]
        return exists, url
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("VOICE CATALOG VALIDATION")
    print("=" * 80)
    
    total_voices = 0
    valid_voices = 0
    invalid_voices = []
    
    for lang_code, voices in sorted(ALL_VOICES.items()):
        print(f"\n{lang_code}: Checking {len(voices)} voices...")
        for voice in voices:
            total_voices += 1
            exists, info = check_voice_url(voice)
            
            if exists:
                valid_voices += 1
                print(f"  ✓ {voice}")
            else:
                invalid_voices.append((voice, info))
                print(f"  ✗ {voice}")
                print(f"    URL: {info}")
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total voices: {total_voices}")
    print(f"Valid voices: {valid_voices}")
    print(f"Invalid voices: {len(invalid_voices)}")
    print(f"Success rate: {(valid_voices/total_voices*100):.1f}%")
    
    if invalid_voices:
        print("\n" + "=" * 80)
        print("INVALID VOICES (need correction):")
        print("=" * 80)
        for voice, info in invalid_voices:
            print(f"  {voice}")
            if info.startswith("http"):
                print(f"    Tried: {info}")
    
    return len(invalid_voices) == 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
