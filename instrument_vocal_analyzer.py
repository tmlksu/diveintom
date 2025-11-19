"""
Instrument and vocal analysis using source separation.
"""
import os
import tempfile
import subprocess
import librosa
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_models import InstrumentInfo, VocalInfo


class InstrumentVocalAnalyzer:
    """Analyzes instruments and vocals using source separation."""

    def __init__(self, audio_path: str, use_separation: bool = True):
        """
        Initialize the analyzer.

        Args:
            audio_path: Path to the audio file
            use_separation: Whether to use source separation (requires demucs)
        """
        self.audio_path = audio_path
        self.use_separation = use_separation
        self.separated_dir = None
        self.stems = {}  # Dictionary to store separated audio stems

        if use_separation:
            self._separate_sources()

    def _separate_sources(self):
        """Separate audio into stems using demucs."""
        try:
            # Create temporary directory for separated files
            self.separated_dir = tempfile.mkdtemp(prefix='music_analysis_')

            # Run demucs
            # Using htdemucs model with 4 stems: vocals, drums, bass, other
            cmd = [
                'demucs',
                '--two-stems=vocals',  # First separate vocals
                '-n', 'htdemucs',
                '-o', self.separated_dir,
                self.audio_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            # Load separated stems
            audio_name = Path(self.audio_path).stem
            model_dir = Path(self.separated_dir) / 'htdemucs' / audio_name

            # Try to load stems
            for stem_name in ['vocals', 'no_vocals']:
                stem_path = model_dir / f'{stem_name}.wav'
                if stem_path.exists():
                    y, sr = librosa.load(str(stem_path))
                    self.stems[stem_name] = (y, sr)

            # Now separate instruments from no_vocals
            if 'no_vocals' in self.stems:
                # Run demucs again for instrument separation
                no_vocals_path = model_dir / 'no_vocals.wav'
                cmd = [
                    'demucs',
                    '-n', 'htdemucs',
                    '-o', self.separated_dir,
                    str(no_vocals_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Load instrument stems
                inst_dir = Path(self.separated_dir) / 'htdemucs' / 'no_vocals'
                for stem_name in ['drums', 'bass', 'other']:
                    stem_path = inst_dir / f'{stem_name}.wav'
                    if stem_path.exists():
                        y, sr = librosa.load(str(stem_path))
                        self.stems[stem_name] = (y, sr)

        except Exception as e:
            print(f"Warning: Source separation failed: {e}")
            print("Continuing with analysis on full mix...")
            self.use_separation = False

    def analyze_vocals(
        self,
        start_time: float,
        end_time: float,
        full_audio: Optional[Tuple[np.ndarray, int]] = None
    ) -> VocalInfo:
        """
        Analyze vocal characteristics in a segment.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            full_audio: Optional tuple of (audio_data, sample_rate) for full mix

        Returns:
            VocalInfo object
        """
        if self.use_separation and 'vocals' in self.stems:
            y_vocals, sr = self.stems['vocals']
        elif full_audio is not None:
            y_vocals, sr = full_audio
        else:
            # Load full audio
            y_vocals, sr = librosa.load(self.audio_path)

        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_segment = y_vocals[start_sample:end_sample]

        if len(y_segment) == 0:
            return VocalInfo(present=False)

        # Check if vocals are present (energy threshold)
        rms = librosa.feature.rms(y=y_segment)[0]
        energy = np.mean(rms)

        if energy < 0.01:  # Low energy threshold
            return VocalInfo(present=False)

        # Extract pitch for gender estimation
        pitch_range = self._extract_pitch_range(y_segment, sr)

        # Estimate gender based on pitch
        gender = self._estimate_gender(pitch_range)

        # Estimate singing style
        singing_style = self._estimate_singing_style(y_segment, sr)

        # Calculate vocal energy
        vocal_energy = float(min(1.0, energy * 20))

        return VocalInfo(
            present=True,
            gender=gender,
            pitch_range=pitch_range,
            singing_style=singing_style,
            vocal_energy=vocal_energy
        )

    def analyze_instruments(
        self,
        start_time: float,
        end_time: float
    ) -> List[InstrumentInfo]:
        """
        Analyze instruments in a segment.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of InstrumentInfo objects
        """
        instruments = []

        if self.use_separation:
            # Analyze separated stems
            instrument_stems = {
                'drums': 'drums',
                'bass': 'bass',
                'other': 'other instruments'  # Could be guitar, piano, synth, etc.
            }

            for stem_key, inst_name in instrument_stems.items():
                if stem_key in self.stems:
                    y, sr = self.stems[stem_key]
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    y_segment = y[start_sample:end_sample]

                    if len(y_segment) == 0:
                        continue

                    # Check if instrument is present
                    rms = librosa.feature.rms(y=y_segment)[0]
                    energy = np.mean(rms)

                    if energy > 0.01:  # Threshold for presence
                        pitch_range = self._extract_pitch_range(y_segment, sr)
                        technique = self._estimate_technique(y_segment, sr, stem_key)
                        volume_level = float(min(1.0, energy * 10))

                        instruments.append(InstrumentInfo(
                            name=inst_name,
                            confidence=0.7,  # Moderate confidence from separation
                            pitch_range=pitch_range if pitch_range['mean'] > 0 else None,
                            playing_technique=technique,
                            volume_level=volume_level
                        ))
        else:
            # Without separation, do basic analysis
            y, sr = librosa.load(self.audio_path)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y_segment = y[start_sample:end_sample]

            if len(y_segment) > 0:
                # Detect presence of percussion
                if self._has_percussion(y_segment, sr):
                    instruments.append(InstrumentInfo(
                        name='percussion/drums',
                        confidence=0.5,
                        playing_technique='rhythmic',
                        volume_level=0.6
                    ))

                # General harmonic instruments
                pitch_range = self._extract_pitch_range(y_segment, sr)
                if pitch_range['mean'] > 0:
                    instruments.append(InstrumentInfo(
                        name='harmonic instruments',
                        confidence=0.4,
                        pitch_range=pitch_range,
                        playing_technique='sustained',
                        volume_level=0.5
                    ))

        return instruments

    def _extract_pitch_range(
        self,
        y: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract pitch range from audio segment."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0}

        return {
            'min': float(np.min(pitch_values)),
            'max': float(np.max(pitch_values)),
            'mean': float(np.mean(pitch_values)),
        }

    def _estimate_gender(self, pitch_range: Dict[str, float]) -> str:
        """
        Estimate vocal gender based on pitch.

        Typical ranges:
        - Male: 85-180 Hz (fundamental)
        - Female: 165-255 Hz (fundamental)
        """
        mean_pitch = pitch_range['mean']

        if mean_pitch == 0:
            return 'unknown'

        if mean_pitch < 165:
            return 'male'
        elif mean_pitch > 200:
            return 'female'
        else:
            return 'mixed/androgynous'

    def _estimate_singing_style(self, y: np.ndarray, sr: int) -> str:
        """Estimate singing style based on audio characteristics."""
        # Calculate dynamic range
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms) - np.min(rms)

        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = np.mean(onset_env)

        # Zero crossing rate (can indicate breathiness/whisper)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)

        # Heuristic classification
        if dynamic_range > 0.3 and onset_strength > 0.5:
            return 'powerful/belting'
        elif zcr_mean > 0.15:
            return 'breathy/whisper'
        elif onset_strength > 0.4:
            return 'rhythmic/rap'
        elif dynamic_range < 0.1:
            return 'soft/gentle'
        else:
            return 'balanced/melodic'

    def _estimate_technique(
        self,
        y: np.ndarray,
        sr: int,
        instrument_type: str
    ) -> str:
        """Estimate playing technique for an instrument."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = np.mean(onset_env)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = np.mean(spectral_centroid)

        if instrument_type == 'drums':
            if onset_strength > 1.0:
                return 'heavy/loud'
            else:
                return 'light/moderate'

        elif instrument_type == 'bass':
            if onset_strength > 0.8:
                return 'plucked/slap'
            else:
                return 'sustained/fingered'

        elif instrument_type == 'other':
            # Could be guitar, piano, synth, etc.
            if onset_strength > 0.7:
                if brightness > 2000:
                    return 'strumming/bright'
                else:
                    return 'plucked/staccato'
            else:
                if brightness > 3000:
                    return 'sustained/bright (synth/strings)'
                else:
                    return 'sustained/warm (piano/pad)'

        return 'unknown'

    def _has_percussion(self, y: np.ndarray, sr: int) -> bool:
        """Detect if percussion is present."""
        # Use onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        # If there are regular onsets, likely percussion
        return len(onset_frames) > 10

    def cleanup(self):
        """Clean up temporary files."""
        if self.separated_dir and os.path.exists(self.separated_dir):
            import shutil
            try:
                shutil.rmtree(self.separated_dir)
            except:
                pass
