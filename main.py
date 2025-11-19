"""
Main orchestrator for music mood analysis.
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from audio_analyzer import AudioAnalyzer
from instrument_vocal_analyzer import InstrumentVocalAnalyzer
from llm_analyzer import LLMAnalyzer
from data_models import (
    SongAnalysis,
    LyricSegment,
    MusicalFeatures,
    MoodInterpretation
)


class MusicMoodAnalyzer:
    """Main orchestrator for music mood analysis."""

    def __init__(
        self,
        audio_path: str,
        lyrics_path: Optional[str] = None,
        use_llm: bool = True,
        use_separation: bool = True,
        llm_provider: str = "gemini",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the analyzer.

        Args:
            audio_path: Path to the MP3/audio file
            lyrics_path: Path to the lyrics text file
            use_llm: Whether to use LLM for mood interpretation
            use_separation: Whether to use source separation for instruments/vocals
            llm_provider: LLM provider - "claude" or "gemini" (default: "gemini")
            llm_model: LLM model to use (optional, uses provider default)
            api_key: API key for LLM analysis (provider-specific)
        """
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path
        self.use_llm = use_llm
        self.use_separation = use_separation

        # Initialize analyzers
        print("Loading audio...")
        self.audio_analyzer = AudioAnalyzer(audio_path)

        print("Initializing instrument/vocal analyzer...")
        self.iv_analyzer = InstrumentVocalAnalyzer(
            audio_path,
            use_separation=use_separation
        )

        if use_llm:
            print(f"Initializing LLM analyzer (provider: {llm_provider})...")
            self.llm_analyzer = LLMAnalyzer(
                provider=llm_provider,
                api_key=api_key,
                model=llm_model
            )
        else:
            self.llm_analyzer = None

        # Load lyrics
        self.lyrics = self._load_lyrics()

    def _load_lyrics(self) -> List[str]:
        """Load lyrics from file."""
        if not self.lyrics_path:
            return []

        with open(self.lyrics_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            # Filter out empty lines
            return [line for line in lines if line]

    def analyze(
        self,
        time_per_line: Optional[float] = None
    ) -> SongAnalysis:
        """
        Perform complete analysis of the song.

        Args:
            time_per_line: Optional fixed time per lyric line in seconds.
                          If None, will divide evenly based on song duration.

        Returns:
            SongAnalysis object with complete analysis
        """
        duration = self.audio_analyzer.duration
        overall_tempo = self.audio_analyzer.get_tempo()
        overall_key, overall_mode = self.audio_analyzer.get_key_and_mode()

        print(f"\nSong duration: {duration:.1f}s")
        print(f"Overall tempo: {overall_tempo:.1f} BPM")
        print(f"Overall key: {overall_key} {overall_mode}")

        # Calculate time segments
        if not self.lyrics:
            print("No lyrics provided. Analyzing full song as one segment.")
            segments = [(0.0, duration, "Instrumental")]
        else:
            # Divide song into segments based on lyrics
            num_lines = len(self.lyrics)

            if time_per_line is None:
                time_per_line = duration / num_lines

            segments = []
            for i, lyric in enumerate(self.lyrics):
                start = i * time_per_line
                end = min((i + 1) * time_per_line, duration)
                segments.append((start, end, lyric))

        # Analyze each segment
        print(f"\nAnalyzing {len(segments)} segments...")
        lyric_segments = []

        for i, (start, end, lyric) in enumerate(tqdm(segments, desc="Processing segments")):
            segment_analysis = self._analyze_segment(
                line_number=i + 1,
                lyric_text=lyric,
                start_time=start,
                end_time=end
            )
            lyric_segments.append(segment_analysis)

        # Overall mood analysis
        overall_mood = None
        if self.llm_analyzer:
            print("\nGenerating overall mood analysis...")
            segment_dicts = [seg.model_dump() for seg in lyric_segments]
            overall_mood = self.llm_analyzer.analyze_overall_song(
                segment_dicts,
                title=Path(self.audio_path).stem
            )

        # Create final analysis object
        analysis = SongAnalysis(
            title=Path(self.audio_path).stem,
            duration=duration,
            overall_tempo=overall_tempo,
            overall_key=f"{overall_key} {overall_mode}",
            lyric_segments=lyric_segments,
            overall_mood=overall_mood,
            analysis_metadata={
                'audio_file': self.audio_path,
                'lyrics_file': self.lyrics_path,
                'used_llm': self.use_llm,
                'used_source_separation': self.use_separation,
                'num_segments': len(segments)
            }
        )

        return analysis

    def _analyze_segment(
        self,
        line_number: int,
        lyric_text: str,
        start_time: float,
        end_time: float
    ) -> LyricSegment:
        """Analyze a single lyric segment."""

        # Musical features
        music_features_dict = self.audio_analyzer.analyze_segment(start_time, end_time)
        musical_features = MusicalFeatures(**music_features_dict)

        # Instruments
        instruments = self.iv_analyzer.analyze_instruments(start_time, end_time)

        # Vocals
        vocal_info = self.iv_analyzer.analyze_vocals(start_time, end_time)

        # LLM interpretation
        mood_interpretation = None
        if self.llm_analyzer:
            mood_interpretation = self.llm_analyzer.analyze_segment(
                lyric_text=lyric_text,
                musical_features=musical_features,
                instruments=instruments,
                vocal_info=vocal_info
            )

        return LyricSegment(
            line_number=line_number,
            lyric_text=lyric_text,
            start_time=start_time,
            end_time=end_time,
            musical_features=musical_features,
            instruments=instruments,
            vocal_info=vocal_info,
            mood_interpretation=mood_interpretation
        )

    def cleanup(self):
        """Clean up temporary files."""
        self.iv_analyzer.cleanup()


def save_output(analysis: SongAnalysis, output_path: str, format: str = 'json'):
    """
    Save analysis to file.

    Args:
        analysis: SongAnalysis object
        output_path: Output file path
        format: 'json' or 'yaml'
    """
    data = analysis.model_dump(exclude_none=False)

    if format == 'yaml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    else:  # json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nAnalysis saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze music mood and generate structured output'
    )
    parser.add_argument(
        'audio_file',
        help='Path to MP3/audio file'
    )
    parser.add_argument(
        '--lyrics',
        '-l',
        help='Path to lyrics text file (one line per lyric line)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: <audio_name>_analysis.json)',
        default=None
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['json', 'yaml'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM-based mood interpretation'
    )
    parser.add_argument(
        '--no-separation',
        action='store_true',
        help='Disable source separation (faster but less detailed)'
    )
    parser.add_argument(
        '--time-per-line',
        type=float,
        default=None,
        help='Fixed time per lyric line in seconds (default: auto-calculate)'
    )
    parser.add_argument(
        '--provider',
        '-p',
        choices=['claude', 'gemini'],
        default='gemini',
        help='LLM provider: claude or gemini via OpenRouter (default: gemini)'
    )
    parser.add_argument(
        '--model',
        '-m',
        help='LLM model to use (default: provider-specific default)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for LLM provider (ANTHROPIC_API_KEY or OPENROUTER_API_KEY env var)'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        audio_stem = Path(args.audio_file).stem
        ext = 'yaml' if args.format == 'yaml' else 'json'
        output_path = f"{audio_stem}_analysis.{ext}"
    else:
        output_path = args.output

    # Create analyzer
    analyzer = MusicMoodAnalyzer(
        audio_path=args.audio_file,
        lyrics_path=args.lyrics,
        use_llm=not args.no_llm,
        use_separation=not args.no_separation,
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=args.api_key
    )

    try:
        # Run analysis
        print("\n" + "="*60)
        print("MUSIC MOOD ANALYZER")
        print("="*60)

        analysis = analyzer.analyze(time_per_line=args.time_per_line)

        # Save output
        save_output(analysis, output_path, format=args.format)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

    finally:
        # Cleanup
        analyzer.cleanup()


if __name__ == '__main__':
    main()
