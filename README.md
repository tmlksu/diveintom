# Music Mood Analyzer

音楽のMP3ファイルを分析し、歌詞の行ごとに音楽的特徴、雰囲気、意味を構造化データ（JSON/YAML）として出力するPythonツールです。

## 主な機能

### 🎵 音楽分析
- **テンポ・キー・コード検出**: BPM、調性（メジャー/マイナー）、コード進行
- **エネルギー・音量分析**: 盛り上がりや曲の強弱を時系列で追跡
- **スペクトル分析**: 音の明るさや周波数特性

### 🎸 楽器分析
- **楽器分離**: Demucsを使用した音源分離（ドラム、ベース、その他楽器）
- **演奏技法検出**: ストラミング、プラッキング、サステイン等
- **ピッチ範囲**: 各楽器の音域分析

### 🎤 ボーカル分析
- **性別推定**: 男性/女性/ミックス
- **歌唱スタイル**: パワフル、ソフト、ラップ、ウィスパー等
- **ボーカルエネルギー**: 歌声の強度

### 🤖 LLM による解釈
- **ムード分析**: 楽曲の雰囲気を自然言語で解釈
- **感情抽出**: 表現されている感情を列挙
- **意味解釈**: 音楽的瞬間が何を表現しているか
- **ナラティブ機能**: イントロ、ビルドアップ、クライマックス等の構造分析

### 📊 構造化出力
- **JSON/YAML形式**: LLMでの後続処理に最適な構造化データ
- **歌詞行ごとの詳細分析**: タイムスタンプ付き
- **全体的なムード**: 楽曲全体の雰囲気サマリー

## インストール

### 必要要件
- Python 3.8以上
- FFmpeg（オーディオ処理用）

### セットアップ

1. リポジトリのクローン:
```bash
git clone <repository-url>
cd diveintom
```

2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

3. FFmpegのインストール（未インストールの場合）:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

4. APIキーの設定（LLM分析を使用する場合）:

**Gemini使用（デフォルト、推奨）:**
```bash
cp .env.example .env
# .envファイルを編集してOPENROUTER_API_KEYを設定
# OpenRouter APIキーは https://openrouter.ai/keys から取得
```

または環境変数として設定:
```bash
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Claude使用:**
```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 使い方

### 基本的な使用方法

```bash
python main.py <audio_file> --lyrics <lyrics_file>
```

### 例

```bash
# Gemini使用（デフォルト、JSON形式で出力）
python main.py song.mp3 --lyrics lyrics.txt

# Claudeを使用
python main.py song.mp3 --lyrics lyrics.txt --provider claude

# 特定のモデルを指定
python main.py song.mp3 --lyrics lyrics.txt --provider gemini --model google/gemini-pro

# YAML形式で出力
python main.py song.mp3 --lyrics lyrics.txt --format yaml --output analysis.yaml

# LLMを使わず高速に分析
python main.py song.mp3 --lyrics lyrics.txt --no-llm

# 音源分離を無効化（高速だが詳細度低）
python main.py song.mp3 --lyrics lyrics.txt --no-separation

# 歌詞の各行に固定時間を割り当て
python main.py song.mp3 --lyrics lyrics.txt --time-per-line 5.0
```

### コマンドラインオプション

| オプション | 説明 |
|----------|------|
| `audio_file` | MP3/音声ファイルのパス（必須） |
| `--lyrics`, `-l` | 歌詞テキストファイルのパス（1行＝1フレーズ） |
| `--output`, `-o` | 出力ファイルパス（デフォルト: `<audio_name>_analysis.json`） |
| `--format`, `-f` | 出力形式: `json` または `yaml`（デフォルト: `json`） |
| `--provider`, `-p` | LLMプロバイダー: `claude` または `gemini`（デフォルト: `gemini`） |
| `--model`, `-m` | 使用するLLMモデル（デフォルト: プロバイダー固有のデフォルト） |
| `--no-llm` | LLMによるムード解釈を無効化 |
| `--no-separation` | 音源分離を無効化（高速化） |
| `--time-per-line` | 歌詞の各行に割り当てる固定時間（秒）（デフォルト: 自動計算） |
| `--api-key` | LLMプロバイダーのAPIキー（環境変数でも設定可） |

### 歌詞ファイルの形式

歌詞ファイルは1行1フレーズのプレーンテキストです:

```text
夜空を見上げて
君のことを思う
遠く離れていても
心は繋がってる
```

## 出力形式

### JSON出力例

```json
{
  "title": "song",
  "duration": 180.5,
  "overall_tempo": 120.0,
  "overall_key": "C major",
  "lyric_segments": [
    {
      "line_number": 1,
      "lyric_text": "夜空を見上げて",
      "start_time": 0.0,
      "end_time": 5.0,
      "musical_features": {
        "tempo": 120.0,
        "key": "C",
        "mode": "major",
        "chord": "C",
        "energy": 0.45,
        "loudness": -12.5,
        "spectral_centroid": 2500.0,
        "spectral_rolloff": 5000.0
      },
      "instruments": [
        {
          "name": "guitar",
          "confidence": 0.7,
          "pitch_range": {
            "min": 200.0,
            "max": 800.0,
            "mean": 450.0
          },
          "playing_technique": "fingerpicking",
          "volume_level": 0.6
        }
      ],
      "vocal_info": {
        "present": true,
        "gender": "male",
        "pitch_range": {
          "min": 150.0,
          "max": 300.0,
          "mean": 220.0
        },
        "singing_style": "soft/gentle",
        "vocal_energy": 0.5
      },
      "mood_interpretation": {
        "mood": "contemplative and peaceful",
        "emotions": ["nostalgic", "gentle", "reflective"],
        "intensity": 0.5,
        "meaning": "This opening creates a serene, introspective atmosphere, inviting the listener into a moment of quiet reflection under the night sky.",
        "atmosphere": "soft and dreamy",
        "narrative_function": "intro"
      }
    }
  ],
  "overall_mood": "A contemplative journey through memories and emotions",
  "analysis_metadata": {
    "audio_file": "song.mp3",
    "lyrics_file": "lyrics.txt",
    "used_llm": true,
    "used_source_separation": true,
    "num_segments": 4
  }
}
```

## データモデル

詳細なデータ構造は `data_models.py` を参照してください。主要なモデル:

- **SongAnalysis**: 楽曲全体の分析結果
- **LyricSegment**: 歌詞行ごとの分析
- **MusicalFeatures**: 音楽的特徴（テンポ、キー、エネルギー等）
- **InstrumentInfo**: 楽器情報（種類、奏法、ピッチ等）
- **VocalInfo**: ボーカル情報（性別、スタイル等）
- **MoodInterpretation**: LLMによるムード解釈

## アーキテクチャ

```
main.py
├── AudioAnalyzer (audio_analyzer.py)
│   └── librosaを使用した音楽特徴抽出
├── InstrumentVocalAnalyzer (instrument_vocal_analyzer.py)
│   └── Demucsによる音源分離と楽器/ボーカル分析
└── LLMAnalyzer (llm_analyzer.py)
    └── Claude APIによるムード・意味解釈
```

## パフォーマンスとコスト

### 処理時間
- **音源分離あり**: 3-5分の楽曲で約5-10分
- **音源分離なし**: 3-5分の楽曲で約30秒-1分

### LLMコスト
- **Gemini 2.0 Flash（デフォルト）**: 無料（OpenRouterの無料ティア使用時）
- **Claude 3.5 Haiku**: 約 $0.01-0.05 per song（歌詞の行数による）
- **Gemini Pro（有料）**: 約 $0.002-0.01 per song

**推奨**: コスト削減のため、デフォルトのGemini無料ティアの使用をお勧めします。

## トラブルシューティング

### FFmpegエラー
```
Error: FFmpeg not found
```
→ FFmpegをインストールしてください（上記「インストール」セクション参照）

### Demucsエラー
```
Source separation failed
```
→ `--no-separation` オプションを使用して音源分離を無効化できます

### APIキーエラー

**Gemini使用時:**
```
ValueError: OpenRouter API key required
```
→ `.env`ファイルまたは環境変数で`OPENROUTER_API_KEY`を設定してください
→ APIキーは https://openrouter.ai/keys から取得できます

**Claude使用時:**
```
ValueError: API key required
```
→ `.env`ファイルまたは環境変数で`ANTHROPIC_API_KEY`を設定してください

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！

## 今後の改善予定

- [ ] より高度な楽器認識（ピアノ、バイオリン等の詳細分類）
- [ ] リアルタイム分析モード
- [ ] 可視化機能（グラフ生成）
- [ ] 歌詞の自動タイムスタンプ同期
- [ ] WebUI
