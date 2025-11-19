"""
LLM-based mood and meaning analysis.
"""
import os
import json
from typing import Dict, List, Optional
from anthropic import Anthropic

from data_models import MoodInterpretation, MusicalFeatures, InstrumentInfo, VocalInfo


class LLMAnalyzer:
    """Uses LLM to interpret musical features and generate mood analysis."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize the LLM analyzer.

        Args:
            api_key: Anthropic API key (if None, will use ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-3-5-haiku for cost efficiency)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def analyze_segment(
        self,
        lyric_text: str,
        musical_features: MusicalFeatures,
        instruments: List[InstrumentInfo],
        vocal_info: VocalInfo,
        context: Optional[str] = None
    ) -> MoodInterpretation:
        """
        Analyze a music segment and generate mood interpretation.

        Args:
            lyric_text: The lyric text for this segment
            musical_features: Musical features extracted from audio
            instruments: List of detected instruments
            vocal_info: Vocal characteristics
            context: Optional context about the song or previous segments

        Returns:
            MoodInterpretation object
        """
        # Build prompt
        prompt = self._build_prompt(
            lyric_text,
            musical_features,
            instruments,
            vocal_info,
            context
        )

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse response
            response_text = response.content[0].text
            interpretation = self._parse_response(response_text)

            return interpretation

        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
            # Return fallback interpretation
            return self._fallback_interpretation(musical_features)

    def _build_prompt(
        self,
        lyric_text: str,
        musical_features: MusicalFeatures,
        instruments: List[InstrumentInfo],
        vocal_info: VocalInfo,
        context: Optional[str] = None
    ) -> str:
        """Build the prompt for LLM analysis."""

        # Format musical features
        music_desc = f"""
Musical Features:
- Tempo: {musical_features.tempo:.1f} BPM
- Key: {musical_features.key or 'Unknown'} {musical_features.mode or ''}
- Chord: {musical_features.chord or 'Unknown'}
- Energy: {musical_features.energy:.2f} (0.0 = low, 1.0 = high)
- Loudness: {musical_features.loudness:.1f} dB
- Spectral Brightness: {musical_features.spectral_centroid or 0:.1f} Hz
"""

        # Format instruments
        if instruments:
            inst_desc = "Instruments:\n"
            for inst in instruments:
                inst_desc += f"- {inst.name}"
                if inst.playing_technique:
                    inst_desc += f" ({inst.playing_technique})"
                if inst.volume_level:
                    inst_desc += f" - Volume: {inst.volume_level:.2f}"
                inst_desc += "\n"
        else:
            inst_desc = "Instruments: Not clearly identified\n"

        # Format vocals
        if vocal_info.present:
            vocal_desc = f"""
Vocals:
- Gender: {vocal_info.gender or 'Unknown'}
- Singing Style: {vocal_info.singing_style or 'Unknown'}
- Vocal Energy: {vocal_info.vocal_energy or 0:.2f}
"""
        else:
            vocal_desc = "Vocals: Instrumental section (no vocals)\n"

        # Context
        context_desc = f"\nContext: {context}\n" if context else ""

        prompt = f"""You are a music analyst. Analyze the following music segment and provide a detailed interpretation.

{context_desc}
Lyrics: "{lyric_text}"

{music_desc}
{inst_desc}
{vocal_desc}

Please provide your analysis in the following JSON format:
{{
    "mood": "primary mood (e.g., energetic, melancholic, peaceful, tense)",
    "emotions": ["emotion1", "emotion2", "emotion3"],
    "intensity": 0.0-1.0,
    "meaning": "what this musical moment represents or conveys (2-3 sentences)",
    "atmosphere": "atmospheric description (e.g., tense, uplifting, dreamy, dark)",
    "narrative_function": "function in the song (e.g., intro, buildup, climax, resolution, verse, chorus, bridge)"
}}

Consider:
1. How the musical features (tempo, key, energy) relate to the lyrics
2. How the instruments and their playing styles contribute to the mood
3. The emotional arc and what this moment means in the song's narrative
4. The overall atmosphere created by combining all elements

Respond ONLY with valid JSON, no additional text.
"""

        return prompt

    def _parse_response(self, response_text: str) -> MoodInterpretation:
        """Parse the LLM response into a MoodInterpretation object."""
        try:
            # Try to extract JSON from response
            # Sometimes the model includes markdown code blocks
            response_text = response_text.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            data = json.loads(response_text)

            return MoodInterpretation(
                mood=data.get('mood', 'neutral'),
                emotions=data.get('emotions', []),
                intensity=float(data.get('intensity', 0.5)),
                meaning=data.get('meaning', ''),
                atmosphere=data.get('atmosphere', ''),
                narrative_function=data.get('narrative_function')
            )

        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            print(f"Response was: {response_text}")

            # Try to extract some info manually
            return MoodInterpretation(
                mood='unknown',
                emotions=[],
                intensity=0.5,
                meaning=response_text[:200] if response_text else 'Analysis unavailable',
                atmosphere='unknown'
            )

    def _fallback_interpretation(
        self,
        musical_features: MusicalFeatures
    ) -> MoodInterpretation:
        """Generate a basic interpretation without LLM."""
        # Simple heuristic-based interpretation
        energy = musical_features.energy
        mode = musical_features.mode or 'unknown'

        if energy > 0.7:
            if mode == 'major':
                mood = 'energetic and uplifting'
                emotions = ['excited', 'joyful']
                atmosphere = 'bright and dynamic'
            else:
                mood = 'intense and powerful'
                emotions = ['intense', 'aggressive']
                atmosphere = 'dark and energetic'
        elif energy > 0.4:
            if mode == 'major':
                mood = 'moderate and positive'
                emotions = ['pleasant', 'balanced']
                atmosphere = 'warm and steady'
            else:
                mood = 'contemplative'
                emotions = ['thoughtful', 'serious']
                atmosphere = 'moody and introspective'
        else:
            if mode == 'major':
                mood = 'calm and peaceful'
                emotions = ['relaxed', 'gentle']
                atmosphere = 'soft and soothing'
            else:
                mood = 'melancholic'
                emotions = ['sad', 'somber']
                atmosphere = 'dark and subdued'

        return MoodInterpretation(
            mood=mood,
            emotions=emotions,
            intensity=energy,
            meaning=f"A {mood} section with {mode} tonality.",
            atmosphere=atmosphere,
            narrative_function='verse'  # Default guess
        )

    def analyze_overall_song(
        self,
        all_segments: List[Dict],
        title: Optional[str] = None
    ) -> str:
        """
        Analyze the overall mood of the entire song.

        Args:
            all_segments: List of all segment analyses
            title: Optional song title

        Returns:
            Overall mood description
        """
        # Build summary prompt
        moods = [seg.get('mood_interpretation', {}).get('mood', 'unknown')
                for seg in all_segments
                if seg.get('mood_interpretation')]

        prompt = f"""Based on the following segment moods from a song{f' titled "{title}"' if title else ''},
provide a concise overall mood description (1-2 sentences):

Segment moods: {', '.join(moods[:10])}

Respond with just the description, no additional formatting."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"Warning: Overall analysis failed: {e}")
            return "Mixed emotional journey"
