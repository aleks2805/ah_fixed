import json
import librosa
import argparse


LABEL_TO_CODE = {
    "intro": "i",
    "verse": "A",
    "pre-chorus": "A",
    "chorus": "B",
    "bridge": "C",
    "inst": "D",
    "silence": "s",
    "outro": "x"
}


def detect_bpm(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    if hasattr(tempo, "__len__"):
        tempo = tempo.item() if tempo.size == 1 else tempo[0]

    return float(tempo)


def get_music_bounds(segments):

    first = None
    last = None

    for s in segments:
        if s["label"] != "silence":
            first = s
            break

    for s in reversed(segments):
        if s["label"] != "silence":
            last = s
            break

    return first["start"], last["end"]


def parse_songformer(json_path, audio_path, quantization=1):

    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    bpm = detect_bpm(audio_path)
    bar_sec = 4 * 60 / bpm

    start, end = get_music_bounds(segments)
    real_duration = end - start

    filtered = [
        s for s in segments
        if s["label"] != "silence"
    ]

    out = []

    for s in filtered:

        dur_sec = s["end"] - s["start"]

        bars = dur_sec / bar_sec
        bars = round(bars * quantization) / quantization
        bars = round(bars)

        if bars <= 0:
            continue

        code = LABEL_TO_CODE.get(
            s["label"],
            "?"
        )

        out.append(
            f"{code}{bars}"
        )

    return {
        "bpm": round(bpm, 2),
        "real_duration_sec": round(real_duration, 2),
        "structure": "".join(out)
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True,
        help="SongFormer output.json"
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="audio file"
    )

    parser.add_argument(
        "--quantization",
        type=int,
        default=1
    )

    args = parser.parse_args()

    result = parse_songformer(
        json_path=args.json,
        audio_path=args.audio,
        quantization=args.quantization
    )

    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))