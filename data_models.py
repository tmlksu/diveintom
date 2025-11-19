"""
Data models for music mood analysis output.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class InstrumentInfo(BaseModel):
    """Information about a detected instrument."""
    name: str = Field(description="Instrument name (e.g., 'guitar', 'piano', 'drums')")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    pitch_range: Optional[Dict[str, float]] = Field(
        default=None,
        description="Pitch range in Hz (min, max, mean)"
    )
    playing_technique: Optional[str] = Field(
        default=None,
        description="Playing technique or style (e.g., 'strumming', 'fingerpicking', 'legato')"
    )
    volume_level: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relative volume level"
    )


class VocalInfo(BaseModel):
    """Information about vocal characteristics."""
    present: bool = Field(description="Whether vocals are detected")
    gender: Optional[str] = Field(
        default=None,
        description="Estimated gender: 'male', 'female', 'mixed', 'unknown'"
    )
    pitch_range: Optional[Dict[str, float]] = Field(
        default=None,
        description="Vocal pitch range in Hz"
    )
    singing_style: Optional[str] = Field(
        default=None,
        description="Singing style (e.g., 'soft', 'powerful', 'rap', 'whisper', 'belting')"
    )
    vocal_energy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Vocal energy/intensity"
    )


class MusicalFeatures(BaseModel):
    """Musical features for a time segment."""
    tempo: float = Field(description="Tempo in BPM")
    key: Optional[str] = Field(
        default=None,
        description="Musical key (e.g., 'C major', 'A minor')"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Mode: 'major', 'minor', or other"
    )
    chord: Optional[str] = Field(
        default=None,
        description="Predominant chord in this segment"
    )
    energy: float = Field(
        ge=0.0,
        le=1.0,
        description="Energy level (0.0 to 1.0)"
    )
    loudness: float = Field(description="Loudness in dB")
    spectral_centroid: Optional[float] = Field(
        default=None,
        description="Spectral centroid (brightness)"
    )
    spectral_rolloff: Optional[float] = Field(
        default=None,
        description="Spectral rolloff frequency"
    )


class MoodInterpretation(BaseModel):
    """LLM-generated mood and meaning interpretation."""
    mood: str = Field(description="Overall mood (e.g., 'energetic', 'melancholic', 'peaceful')")
    emotions: List[str] = Field(
        default_factory=list,
        description="List of emotions conveyed"
    )
    intensity: float = Field(
        ge=0.0,
        le=1.0,
        description="Emotional intensity"
    )
    meaning: str = Field(
        description="What this musical moment represents or conveys"
    )
    atmosphere: str = Field(
        description="Atmospheric description (e.g., 'tense', 'uplifting', 'dreamy')"
    )
    narrative_function: Optional[str] = Field(
        default=None,
        description="Function in the song narrative (e.g., 'intro', 'buildup', 'climax', 'resolution')"
    )


class LyricSegment(BaseModel):
    """Analysis for a single lyric line."""
    line_number: int = Field(description="Line number in lyrics (1-indexed)")
    lyric_text: str = Field(description="The lyric text for this line")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")

    # Musical features
    musical_features: MusicalFeatures

    # Instruments
    instruments: List[InstrumentInfo] = Field(
        default_factory=list,
        description="Detected instruments in this segment"
    )

    # Vocals
    vocal_info: VocalInfo

    # LLM interpretation
    mood_interpretation: Optional[MoodInterpretation] = Field(
        default=None,
        description="LLM-generated mood and meaning interpretation"
    )


class SongAnalysis(BaseModel):
    """Complete song analysis output."""
    title: Optional[str] = Field(default=None, description="Song title")
    duration: float = Field(description="Total duration in seconds")
    overall_tempo: float = Field(description="Overall average tempo")
    overall_key: Optional[str] = Field(
        default=None,
        description="Overall key of the song"
    )

    # Lyric-by-lyric analysis
    lyric_segments: List[LyricSegment] = Field(
        description="Analysis for each lyric line"
    )

    # Global interpretation
    overall_mood: Optional[str] = Field(
        default=None,
        description="Overall mood of the song"
    )
    song_structure: Optional[List[str]] = Field(
        default=None,
        description="Song structure (e.g., ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'])"
    )

    # Metadata
    analysis_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the analysis process"
    )
