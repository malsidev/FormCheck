import argparse
import json
from pathlib import Path

from vision.analysis import analyze_image_file, analyze_video_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Формирующий анализ техники физических упражнений")
    parser.add_argument("--file", required=True, help="Путь к изображению или видео")
    parser.add_argument("--exercise", default="squat", choices=["squat", "push-up", "pushup"], help="Упражнение для анализа")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        result = analyze_image_file(file_path, exercise=args.exercise)
    else:
        result = analyze_video_file(file_path, exercise=args.exercise)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()