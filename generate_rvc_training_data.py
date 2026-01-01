"""
Generate audio samples from selected Piper voice for RVC model training.

RVC (Retrieval-based Voice Conversion) training requires:
- 100+ audio samples for reasonable quality
- Clean audio files in WAV format
- Consistent sample rate (40kHz recommended for RVC)
- Duration of a few seconds to a minute per sample
- Diverse phonetic content for better voice representation

Usage:
    python generate_rvc_training_data.py [--output-dir OUTPUT_DIR] [--num-samples NUM_SAMPLES] [--sample-rate SAMPLE_RATE]
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Optional
import subprocess
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import voice_assistant
sys.path.insert(0, str(Path(__file__).parent))
from voice_assistant import VoiceAssistant, get_assistant

# Diverse text samples for RVC training
# These cover various phonemes, consonant clusters, and prosodic patterns
TRAINING_TEXTS = [
    # Basic phoneme coverage
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    
    # Emotional/prosodic variation
    "I absolutely love this beautiful day!",
    "What? No, that's impossible to believe.",
    "Please, can you help me with this?",
    "Amazing! I never thought that would work.",
    "Honestly, I'm not sure about that decision.",
    
    # Consonant clusters
    "Strange characters and symbols everywhere.",
    "Spring brings flowers and fresh green leaves.",
    "String music creates strong emotional responses.",
    "Strength comes from structured strength training.",
    "Striving for excellence in all endeavors.",
    
    # Vowel variations
    "All about audio amplification and acoustics.",
    "Excellent electronics equipment for everyone.",
    "Idioms illustrate interesting intellectual ideas.",
    "Outstanding opportunities open new horizons.",
    "Understanding urban utilities requires expertise.",
    
    # Difficult phoneme combinations
    "Sixty-six sizzling sausages in the skillet.",
    "Fresh French fries from the freezer.",
    "Pleasure and pressure create leisure treasures.",
    "Weather whether or not we gather together.",
    "The rhythm of the thick thimble.",
    
    # Varied sentence structures
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "Betty Botter bought some butter but the butter was bitter.",
    "Sally sells silly silver slippers slowly.",
    
    # Natural speech patterns
    "I think we should meet tomorrow morning to discuss the project.",
    "Can you believe it's already been a whole year since we started?",
    "The presentation went really well, everyone seemed impressed.",
    "I'm looking forward to the weekend and relaxing a bit.",
    "We need to figure out the best solution for this problem.",
    
    # Longer sentences for prosody
    "The development of artificial intelligence has revolutionized how we work and communicate with each other.",
    "When considering different options, it's important to evaluate the potential benefits and drawbacks.",
    "I really appreciate your help and support throughout this challenging project.",
    "The combination of technology and creativity produces the most innovative results.",
    "Understanding complex concepts requires patience, practice, and persistence.",
    
    # Questions and statements
    "Why do you think that approach would work better?",
    "That's absolutely correct and very well explained.",
    "Could you please elaborate on that particular point?",
    "I completely agree with your assessment of the situation.",
    "What are your thoughts on implementing this new strategy?",
    
    # Additional vowel and consonant coverage
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "Unique and unforgettable experiences create lasting memories.",
    "Reliable research reveals remarkable results repeatedly.",
    "Dull data doesn't describe dynamic decisions well.",
    "Great groups generate genuinely grand gatherings.",
    
    # More consonant clusters
    "Shrewdly shred shredded shrubs and shrunk shrimpshells.",
    "Splendid splashes split across sprawling sprinklers.",
    "Strict structures restrict streaming script strips.",
    "Thrown threads threaten three thrashing thrushes.",
    "Scratchpads scramble through cryptic script scrawls.",
    
    # Additional emotional/prosodic patterns
    "Wow, that's absolutely incredible and wonderful!",
    "Sorry, I didn't quite understand that correctly.",
    "Thank you so much for your incredible patience and help.",
    "Seriously? That seems completely unreasonable to me.",
    "Perhaps we should reconsider our original approach here.",
    "Definitely, that's the best solution we could implement.",
    "Honestly, I'm feeling quite uncertain about this decision.",
    "Wow, I'm impressed with your remarkable dedication.",
    
    # Technical and complex speech
    "Algorithms automatically analyze architectural abstractions adequately.",
    "Methodological frameworks facilitate functional optimization effectively.",
    "Comprehensive computational capabilities compute complex calculations correctly.",
    "Systematic standardization strengthens structural stability significantly.",
    "Innovative infrastructure integrates intelligent integration intelligently.",
    
    # Rapid speech patterns
    "She shed seventeen shimmering silvery shells swiftly.",
    "Quickly quelled questionable quantum quandaries quite quietly.",
    "Brilliantly bounded behavioral biology becomes biological benchmarks.",
    "Swiftly switched streamlined systematic switch statistics swirly.",
    "Precisely presented preemptive principal professional prevention programs.",
    
    # Soft and gentle patterns
    "Whisper softly as we wander through the woods together.",
    "Gently guide the gossamer gift into the garden gazebo.",
    "Hush now, listen to the harmonious humming of the hummingbird.",
    "Please pause and ponder the peaceful path before proceeding.",
    "Meander mindfully, musing on life's meaningful moments.",
    
    # Hard and forceful patterns
    "Concrete construction conquered by catastrophic collapse concerns.",
    "Brutal black backs break through barriers boldly backing.",
    "Crashing, clanging cymbals create chaotic cacophonic calls.",
    "Decisive determination defines determined drivers' destinations determinedly.",
    "Forceful forward focus fortifies formidable fortress foundations.",
    
    # Mixed emotional intensity
    "Listen carefully because this is critically crucial information!",
    "I need to tell you something rather serious and important.",
    "You know what, forget it, it doesn't really matter anyway.",
    "Actually, I think you're absolutely right about everything.",
    "Hold on, let me reconsider that perspective for a moment.",
    
    # Articulation challenges
    "Sixth sheets of sheer silk shimmer in the sunshine.",
    "Theophilus Thistle, the thistle sifter, sifted thistles through a thistle sifter.",
    "Rural juror jury juridical judicious judicial jurisdiction.",
    "Statistics show sixths should suggest systematic solutions.",
    "Worcestershire sauce splashed across the workshop.",
    
    # Varied pacing and rhythm
    "One, two, three, four, five, six, seven, eight, nine, ten.",
    "Slowly, carefully, methodically, we proceed through the process.",
    "Quickly! Rapidly! Fast! Faster! Fastest! Go, go, go!",
    "Tick tock, tick tock, the clock keeps on ticking time away.",
    "Dancing, singing, laughing, playing throughout the entire day.",
    
    # Narrative and storytelling
    "Once upon a time, in a land far away, there lived a brave knight.",
    "The adventure began when we discovered an ancient, mysterious map.",
    "She walked through the forest wondering what adventure awaited her.",
    "As the sun set, golden light painted the sky in brilliant colors.",
    "The ending was unexpected but ultimately satisfying and complete.",
    
    # Formal and professional language
    "I hereby solemnly declare my intention to proceed with the proposal.",
    "Furthermore, the evidence suggests a strong correlation exists.",
    "In conclusion, we recommend immediate implementation of these changes.",
    "The aforementioned factors contribute significantly to our findings.",
    "Consequently, stakeholders must evaluate all available options carefully.",
    
    # Casual and conversational
    "Hey, how's it going? Doing well, thanks for asking!",
    "So like, I was thinking we could totally do something fun.",
    "Yeah, that sounds good to me, I'm definitely on board with that.",
    "Honestly, I've got no clue what she's talking about anymore.",
    "Dude, that's like, literally the craziest thing I've ever heard!",
    
    # Descriptive and sensory language
    "The crisp autumn air smells of cinnamon, apples, and fallen leaves.",
    "Soft velvet fabric feels luxurious and smooth against bare skin.",
    "The bitter taste of dark chocolate melts slowly on the tongue.",
    "Bright, vibrant colors danced across the canvas like living fire.",
    "The soothing sound of ocean waves creates peaceful, rhythmic music.",
    
    # Exclamatory and emphatic
    "Absolutely fantastic! I couldn't have asked for better results!",
    "Seriously?! That's unbelievable! I can't believe it's true!",
    "Oh my goodness! This is incredible! I'm so excited!",
    "What?! No way! That's completely impossible to imagine!",
    "Wow! Amazing! Outstanding! Brilliant! Perfect! Excellent!",
    
    # Questions with various intonations
    "Would you mind helping me with this difficult task?",
    "Don't you think we should reconsider our previous decision?",
    "How exactly would you suggest we approach this problem?",
    "Why on earth would anyone believe such an outlandish claim?",
    "Isn't it obvious that we need to change our strategy now?",
    
    # Abstract and philosophical
    "Time flows like an endless river through the landscape of existence.",
    "Beauty exists not in perfection but in the flaws we embrace.",
    "Knowledge is the light that dispels the darkness of ignorance.",
    "Change is the only constant in this ever-evolving universe.",
    "Truth can be elusive, hiding beneath layers of perception and belief.",
    
    # Scientific and technical terminology
    "Photosynthesis converts solar radiation into chemical energy efficiently.",
    "The quantum mechanics principle demonstrates wave-particle duality conclusively.",
    "Isotopes possess different numbers of neutrons within the nucleus.",
    "Thermodynamics establishes fundamental principles governing energy transfer.",
    "Crystalline structures exhibit periodic patterns at the molecular level.",
    
    # Poetic and lyrical
    "In moonlight dancing, shadows play their timeless waltz of wonder.",
    "Stars like diamonds scattered across velvet canvas of the night.",
    "Morning dew glistens as nature awakens from her peaceful slumber.",
    "Hearts beat in rhythm with the pulse of the eternal universe.",
    "Dreams drift like clouds across the vastness of sleeping minds.",
    
    # Action-oriented and dynamic
    "Run faster! Jump higher! Reach further than ever before!",
    "Attack the problem from multiple angles simultaneously and boldly.",
    "Create solutions that transform challenges into opportunities for growth.",
    "Build momentum through consistent action and unwavering determination.",
    "Accelerate progress by removing obstacles and streamlining processes.",
    
    # Contrasts and comparisons
    "Light and shadow exist in constant eternal tension and balance.",
    "Silence can be louder than words spoken with passionate intensity.",
    "Beginnings often feel like endings viewed from a different perspective.",
    "Simple solutions can sometimes be more effective than complex ones.",
    "Strong and weak are relative terms depending on context and situations.",
]

def load_voice_assistant() -> VoiceAssistant:
    """Load the VoiceAssistant instance with current selected voice."""
    va = get_assistant(robot_controller=None)
    if not va:
        logger.error("Could not initialize VoiceAssistant")
        sys.exit(1)
    return va


def list_available_voices(va: VoiceAssistant) -> List[str]:
    """Get list of available voices from piper directory."""
    va._load_available_voices()
    return va.available_voices


def get_selected_voice_info(va: VoiceAssistant) -> tuple[str, str]:
    """Get the selected voice name and model path."""
    voice_name = va.selected_voice
    if not voice_name:
        logger.error("No voice selected. Please select a voice first.")
        sys.exit(1)
    
    model_path = va.piper_dir / f"{voice_name}.onnx"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Using voice: {voice_name}")
    logger.info(f"Model path: {model_path}")
    
    return voice_name, str(model_path)


def generate_audio_with_piper(
    text: str,
    model_path: str,
    output_file: Path,
    sample_rate: int = 40000
) -> bool:
    """Generate audio for text using Piper CLI.
    
    Args:
        text: Text to synthesize
        model_path: Path to the Piper model (.onnx file)
        output_file: Path where to save the WAV file
        sample_rate: Sample rate (note: Piper uses the voice model's native sample rate, this is informational)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use Piper to generate speech
        # Format: piper --model MODEL_PATH --output_file OUTPUT << TEXT
        # Note: Piper uses the sample rate defined in the voice model's .onnx.json file
        process = subprocess.Popen(
            ["piper", "--model", model_path, "--output_file", str(output_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=text)
        
        if process.returncode != 0:
            logger.warning(f"Piper error: {stderr}")
            return False
        
        if not output_file.exists():
            logger.warning(f"Output file not created: {output_file}")
            return False
        
        file_size = output_file.stat().st_size
        logger.debug(f"Generated audio: {output_file.name} ({file_size} bytes)")
        return True
        
    except FileNotFoundError:
        logger.error("Piper CLI not found. Install with: pip install piper-tts")
        return False
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return False


def generate_training_samples(
    voice_name: str,
    model_path: str,
    output_dir: Path,
    num_samples: Optional[int] = None,
    sample_rate: int = 40000
) -> int:
    """Generate training samples for RVC.
    
    Args:
        voice_name: Name of the selected voice
        model_path: Path to the Piper model
        output_dir: Directory to save training samples
        num_samples: Number of samples to generate (default: use all available texts)
        sample_rate: Sample rate for output audio
        
    Returns:
        Number of successfully generated samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read sample rate from voice model's JSON configuration
    voice_json_path = Path(model_path).parent / f"{Path(model_path).stem}.onnx.json"
    actual_sample_rate = sample_rate  # Default fallback
    
    if voice_json_path.exists():
        try:
            with open(voice_json_path, 'r', encoding='utf-8') as f:
                voice_config = json.load(f)
                if 'audio' in voice_config and 'sample_rate' in voice_config['audio']:
                    actual_sample_rate = voice_config['audio']['sample_rate']
                    logger.info(f"Using voice's configured sample rate: {actual_sample_rate}Hz")
        except Exception as e:
            logger.warning(f"Could not read voice config: {e}, using default: {sample_rate}Hz")
    
    # Create metadata file
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "voice": voice_name,
        "sample_rate": actual_sample_rate,
        "model_path": model_path,
        "total_samples": num_samples or len(TRAINING_TEXTS),
        "samples": []
    }
    
    # Limit samples if specified
    texts = TRAINING_TEXTS[:num_samples] if num_samples else TRAINING_TEXTS
    
    logger.info(f"Generating {len(texts)} training samples...")
    logger.info(f"Output directory: {output_dir}")
    
    successful = 0
    skipped = 0
    for idx, text in enumerate(texts, 1):
        output_file = output_dir / f"sample_{idx:03d}.wav"
        
        # Check if sample already exists
        if output_file.exists():
            logger.info(f"[{idx}/{len(texts)}] Skipping existing sample: {text[:60]}...")
            skipped += 1
            metadata["samples"].append({
                "file": output_file.name,
                "text": text,
                "index": idx
            })
            continue
        
        logger.info(f"[{idx}/{len(texts)}] Generating: {text[:60]}...")
        
        if generate_audio_with_piper(text, model_path, output_file, actual_sample_rate):
            successful += 1
            metadata["samples"].append({
                "file": output_file.name,
                "text": text,
                "index": idx
            })
        else:
            logger.warning(f"Failed to generate sample {idx}")
    
    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nGeneration complete!")
    logger.info(f"Successfully generated: {successful}/{len(texts)} samples")
    if skipped > 0:
        logger.info(f"Skipped existing: {skipped}/{len(texts)} samples")
    logger.info(f"Metadata saved to: {metadata_file}")
    
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio samples from Piper voice for RVC training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/rvc_training_data"),
        help="Output directory for training samples (default: models/rvc_training_data)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=f"Number of samples to generate (default: all {len(TRAINING_TEXTS)})"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=40000,
        help="Sample rate in Hz (default: 40000, recommended for RVC)"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice name to use (e.g., en_US-amy-medium). If not specified, uses selected voice."
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices and exit"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RVC Training Data Generator")
    logger.info("=" * 60)
    
    # Load voice assistant
    logger.info("Loading VoiceAssistant...")
    va = load_voice_assistant()
    
    # Handle voice listing
    if args.list_voices:
        available_voices = list_available_voices(va)
        logger.info(f"\nAvailable voices ({len(available_voices)}):")
        for i, voice in enumerate(available_voices, 1):
            logger.info(f"  {i}. {voice}")
        logger.info("\nUsage: python generate_rvc_training_data.py --voice <voice_name>")
        return
    
    # Determine which voice to use
    if args.voice:
        # Validate requested voice exists
        available_voices = list_available_voices(va)
        if args.voice not in available_voices:
            logger.error(f"Voice '{args.voice}' not found!")
            logger.info(f"\nAvailable voices:")
            for voice in available_voices:
                logger.info(f"  - {voice}")
            sys.exit(1)
        voice_name = args.voice
        model_path = str(va.piper_dir / f"{voice_name}.onnx")
    else:
        # Use selected voice
        voice_name, model_path = get_selected_voice_info(va)
    
    logger.info(f"Using voice: {voice_name}")
    logger.info(f"Model path: {model_path}")
    
    # Generate samples
    successful = generate_training_samples(
        voice_name=voice_name,
        model_path=model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate
    )
    
    logger.info("\n" + "=" * 60)
    if successful > 0:
        logger.info(f"SUCCESS: Generated {successful} training samples")
        logger.info(f"\nNext steps for RVC training:")
        logger.info(f"1. Download RVC from: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
        logger.info(f"2. Create a new RVC model")
        logger.info(f"3. Upload audio samples from: {args.output_dir}")
        logger.info(f"4. Follow RVC training procedure in their documentation")
    else:
        logger.error("FAILED: No samples generated")
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
