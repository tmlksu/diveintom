"""
Audio analysis module for extracting musical features from audio files.
"""
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioAnalyzer:
    """Analyzes audio files to extract musical features."""

    def __init__(self, audio_path: str, sr: int = 22050):
        """
        Initialize the audio analyzer.

        Args:
            audio_path: Path to the audio file
            sr: Sample rate (default: 22050 Hz)
        """
        self.audio_path = audio_path
        self.sr = sr
        self.y = None
        self.duration = None
        self._load_audio()

    def _load_audio(self):
        """Load the audio file."""
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

    def get_tempo(self) -> float:
        """
        Extract tempo (BPM) from the audio.

        Returns:
            Tempo in beats per minute
        """
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return float(tempo)

    def get_key_and_mode(self) -> Tuple[str, str]:
        """
        Estimate the key and mode of the audio.

        Returns:
            Tuple of (key, mode) e.g., ('C', 'major')
        """
        # Use chromagram to estimate key
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)

        # Average over time
        chroma_mean = np.mean(chroma, axis=1)

        # Find the most prominent note
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]

        # Estimate mode (major/minor) based on chord profile
        # This is a simplified approach
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        # Roll to match the detected key
        major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_mean)[0, 1]
        minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_mean)[0, 1]

        mode = 'major' if major_corr > minor_corr else 'minor'

        return key, mode

    def analyze_segment(
        self,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Analyze a specific time segment of the audio.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Dictionary containing musical features
        """
        # Convert time to samples
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        # Extract segment
        y_segment = self.y[start_sample:end_sample]

        if len(y_segment) == 0:
            return self._empty_features()

        # Tempo for this segment
        try:
            tempo, beats = librosa.beat.beat_track(y=y_segment, sr=self.sr)
        except:
            tempo = self.get_tempo()  # Fall back to overall tempo

        # Energy (RMS)
        rms = librosa.feature.rms(y=y_segment)[0]
        energy = float(np.mean(rms))
        energy_normalized = min(1.0, energy * 10)  # Normalize to 0-1

        # Loudness
        loudness = float(librosa.amplitude_to_db(rms, ref=np.max).mean())

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y_segment, sr=self.sr)[0]
        spectral_centroid = float(np.mean(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=self.sr)[0]
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))

        # Chroma for chord estimation
        chroma = librosa.feature.chroma_cqt(y=y_segment, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Estimate chord (simplified)
        chord = self._estimate_chord(chroma_mean)

        # Key estimation for this segment
        key, mode = self._estimate_key_from_chroma(chroma_mean)

        return {
            'tempo': float(tempo),
            'key': key,
            'mode': mode,
            'chord': chord,
            'energy': energy_normalized,
            'loudness': loudness,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff_mean,
        }

    def _estimate_chord(self, chroma_mean: np.ndarray) -> str:
        """
        Estimate the chord from chromagram.

        Args:
            chroma_mean: Mean chroma vector

        Returns:
            Chord name (e.g., 'C', 'Am', 'G7')
        """
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Find root note
        root_idx = np.argmax(chroma_mean)
        root = notes[root_idx]

        # Check if it's major or minor based on the third
        # Major third is 4 semitones up, minor third is 3 semitones up
        third_major_idx = (root_idx + 4) % 12
        third_minor_idx = (root_idx + 3) % 12

        if chroma_mean[third_major_idx] > chroma_mean[third_minor_idx]:
            return root  # Major chord
        else:
            return root + 'm'  # Minor chord

    def _estimate_key_from_chroma(self, chroma_mean: np.ndarray) -> Tuple[str, str]:
        """Estimate key and mode from chroma vector."""
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        key_idx = np.argmax(chroma_mean)
        key = keys[key_idx]

        # Major/minor detection
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_mean)[0, 1]
        minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_mean)[0, 1]

        mode = 'major' if major_corr > minor_corr else 'minor'

        return key, mode

    def _empty_features(self) -> Dict:
        """Return empty features for invalid segments."""
        return {
            'tempo': 0.0,
            'key': None,
            'mode': None,
            'chord': None,
            'energy': 0.0,
            'loudness': -60.0,
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
        }

    def extract_pitch_range(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, float]:
        """
        Extract pitch range from a segment.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Dictionary with min, max, and mean pitch in Hz
        """
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        y_segment = self.y[start_sample:end_sample]

        if len(y_segment) == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0}

        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y_segment, sr=self.sr)

        # Get pitches with significant magnitude
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

    def get_onset_strength(
        self,
        start_time: float,
        end_time: float
    ) -> float:
        """
        Get onset strength (can indicate playing technique).

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Onset strength value
        """
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        y_segment = self.y[start_sample:end_sample]

        if len(y_segment) == 0:
            return 0.0

        onset_env = librosa.onset.onset_strength(y=y_segment, sr=self.sr)
        return float(np.mean(onset_env))
