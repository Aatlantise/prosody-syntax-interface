# CANDOR Data Processing Pipeline

This document explains how the three main Python scripts work together to process conversational data and extract features for analysis.

## Overview

The pipeline processes conversational audio data (CANDOR dataset) to create a rich feature set combining:
- **Linguistic features**: Word-level surprisal values from language models
- **Audio features**: Pitch, intensity, duration, pauses
- **Visual features**: Gaze patterns (speaker and listener)
- **Interaction features**: Backchannels, turn-taking
- **Phonetic features**: Syllable counts, pitch transformations

## Pipeline Stages

### Stage 1: Compute Surprisals
**Script**: `src/get_surprisals_candor.py`

**Purpose**: Computes word-level surprisal values using language models (e.g., GPT-2)

**Inputs**:
- `--in_csv`: Transcript CSV with turns (columns: text, start, stop, speaker)
- `--lm_model`: Language model name (e.g., "gpt2")
- `--context_len`: Number of previous turns to include as context
- `--turn_strategy`: Turn segmentation strategy (e.g., "backbiter")
- `--convo_id`: Conversation identifier

**Process**:
1. Reads turn-based transcript data
2. Prepends context from previous turns (based on `context_len`)
3. Truncates to model's max token limit
4. Computes surprisal for each word using the language model
5. Explodes data to one row per word

**Outputs**:
- CSV with columns: `turn_id`, `turn_start`, `turn_stop`, `speaker`, `word`, `surprisal`, `window_wc`, `context_wc`

---

### Stage 2: Add Audio-Visual Features
**Script**: `src/process_av_features_candor.py`

**Purpose**: Enriches surprisal data with audio, visual, and interaction features

**Inputs**:
- `--surprisals_csv`: Output from Stage 1
- `--transcript_json`: Raw transcript JSON with word-level timestamps
- `--metadata_json`: Metadata with speaker information
- `--turns_csv`: Turn-based transcript CSV
- `--audiophile_csv`: Detailed utterance-level transcript
- `--av_features_csv`: Audio-video features (gaze, intensity)
- `--sound_file`: Audio file (MP3)
- `--convo_id`: Conversation identifier

**Process**:
1. **Align timestamps**: Matches surprisal words with timestamped transcript words
   - Handles punctuation differences
   - Accounts for backchannel words
   - Adds word start/stop times and durations

2. **Extract audio features** (using Parselmouth):
   - Pitch: mean, max, min per word (separate channels)
   - Intensity: mean, max, min per word (separate channels)

3. **Add visual features**:
   - Speaker gaze (`gaze_on`)
   - Listener gaze (`gaze_on_other`)
   - Video intensity

4. **Detect backchannels**:
   - Identifies overlapping backchannel utterances
   - Adds backchannel offset timing
   - Records backchannel utterance text

**Outputs**:
- CSV with all previous columns plus:
  - `word_start`, `word_stop`, `duration`
  - `pre_word_pause`, `post_word_pause`
  - `mean_pitch`, `max_pitch`, `min_pitch`
  - `mean_intensity`, `max_intensity`, `min_intensity`
  - `gaze_on`, `gaze_on_other`
  - `backchannel_overlap`, `backchannel_utterance`, `backchannel_offset`
  - `channel`, `transcript_name`

---

### Stage 3: Add Phonetic Features
**Script**: `src/postprocess_candor.py`

**Purpose**: Adds phonetic features and derived pitch metrics

**Inputs**:
- `--surp_csv`: Output from Stage 2 (aggregated across conversations)
- `--out_csv`: Output file path

**Process**:
1. **Pitch transformations** (converts Hz to semitones above 50 Hz):
   - `max_pitch_semitones_above_50_hz`
   - `min_pitch_semitones_above_50_hz`
   - `mean_pitch_semitones_above_50_hz`

2. **Pitch deltas** (within-turn changes):
   - `delta_max_pitch_semitones_above_50_hz`: Change in max pitch from previous word
   - `delta_range_pitch_semitones`: Change in pitch range
   - `delta_range_pitch_semitones_per_second`: Change in rate of pitch change

3. **Pitch range metrics**:
   - `range_pitch_semitones`: Pitch range within word
   - `range_pitch_semitones_per_second`: Pitch range normalized by duration

4. **Syllable counting**:
   - Uses CMU dictionary for known words
   - Falls back to pyphen hyphenation for unknown words
   - Adds `syllable_count` column

**Outputs**:
- CSV with all previous columns plus phonetic features

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│  • Transcript CSV (turns)                                        │
│  • Transcript JSON (word timestamps)                            │
│  • Metadata JSON (speakers)                                    │
│  • Audio file (MP3)                                            │
│  • Audio-video features CSV                                    │
│  • Audiophile CSV (detailed utterances)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: get_surprisals_candor.py                              │
│  ────────────────────────────────────────────────────────────  │
│  Input: Transcript CSV                                          │
│  Process:                                                       │
│    • Add context from previous turns                            │
│    • Compute word-level surprisal (LM)                         │
│    • Explode to word-level rows                                 │
│  Output: surprisal_{lm_model}_{context_len}_{convo_id}.csv    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: process_av_features_candor.py                         │
│  ────────────────────────────────────────────────────────────  │
│  Input: Surprisal CSV + All raw data sources                   │
│  Process:                                                       │
│    • Align word timestamps                                      │
│    • Extract audio features (pitch, intensity)                 │
│    • Add visual features (gaze)                                 │
│    • Detect backchannels                                       │
│  Output: features_{lm_model}_{context_len}_{convo_id}.csv     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  AGGREGATION     │
                    │  Combine all     │
                    │  conversations   │
                    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: postprocess_candor.py                                 │
│  ────────────────────────────────────────────────────────────  │
│  Input: Aggregated features CSV                                 │
│  Process:                                                       │
│    • Convert pitch to semitones                                 │
│    • Compute pitch deltas and ranges                            │
│    • Count syllables                                            │
│  Output: candor_merged_data_{lm_model}_{context_len}.csv      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  FINAL DATASET   │
                    │  Ready for       │
                    │  Statistical     │
                    │  Analysis       │
                    └──────────────────┘
```

## Key Dependencies

1. **Stage 2 depends on Stage 1**: `process_av_features_candor.py` requires the surprisal CSV output
2. **Stage 3 depends on Stage 2**: `postprocess_candor.py` requires the features CSV
3. **Aggregation happens between Stage 2 and 3**: Multiple conversation outputs are combined before phonetic processing

## Usage Example

```bash
# Stage 1: Compute surprisals
python src/get_surprisals_candor.py \
    --in_csv data/candor/{convo_id}/transcription/transcript_backbiter.csv \
    --out_csv output/candor/surprisal/gpt2/2/{convo_id}/backbiter/surprisal_gpt2_2_backbiter_{convo_id}.csv \
    --lm_model gpt2 \
    --context_len 2 \
    --turn_strategy backbiter \
    --convo_id {convo_id}

# Stage 2: Add audio-visual features
python src/process_av_features_candor.py \
    --surprisals_csv output/candor/surprisal/gpt2/2/{convo_id}/backbiter/surprisal_gpt2_2_backbiter_{convo_id}.csv \
    --transcript_json data/candor/{convo_id}/transcription/transcribe_output.json \
    --metadata_json data/candor/{convo_id}/metadata.json \
    --turns_csv data/candor/{convo_id}/transcription/transcript_backbiter.csv \
    --audiophile_csv data/candor/{convo_id}/transcription/transcript_audiophile.csv \
    --av_features_csv data/candor/{convo_id}/audio_video_features.csv \
    --sound_file data/candor/{convo_id}/processed/{convo_id}.mp3 \
    --surprisals_features_csv output/candor/features/gpt2/2/{convo_id}/backbiter/features_gpt2_2_backbiter_{convo_id}.csv \
    --convo_id {convo_id}

# Stage 3: Add phonetic features (after aggregation)
python src/postprocess_candor.py \
    --surp_csv output/candor/features/gpt2/2/agg_explore/backbiter/features_gpt2_2_backbiter_agg_explore.csv \
    --out_csv output/candor/merged/gpt2/2/backbiter/candor_merged_data_gpt2_2_backbiter_explore.csv
```
