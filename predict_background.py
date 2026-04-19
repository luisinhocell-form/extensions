from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import xgboost as xgb
from PIL import Image, UnidentifiedImageError

from dark_mode_manwha.services.image_processor import ImageProcessor, InvalidImageError, RegionMetrics

DEFAULT_MODEL_PATH = Path("train/artifacts/xgboost/model.json")
DEFAULT_METRICS_PATH = Path("train/artifacts/xgboost/metrics.json")
DEFAULT_DATASET_CSV_PATH = Path("dataset/region_dataset.csv")
DEFAULT_INPUT_DIR = Path("dataset/img")
DEFAULT_OUTPUT_DIR = Path("train/artifacts/xgboost_examples")
DEFAULT_PATTERN = "*.jpg"
DEFAULT_FEATURE_COLUMNS = [
    "area",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "aspect_ratio",
    "centroid_x",
    "centroid_y",
    "mean_intensity",
    "intensity_variance",
    "mean_saturation",
    "saturation_variance",
    "fill_ratio",
    "touches_border",
    "border_touch_ratio",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
]
DEFAULT_MIN_REGION_PIXELS = 100
DEFAULT_WHITE_THRESHOLD = 250


@dataclass(frozen=True)
class PredictedRegion:
    region_id: int
    probability: float
    predicted_label: int
    bounds: tuple[int, int, int, int]
    metrics: RegionMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica o modelo XGBoost e pinta de preto regioes previstas como fundo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Caminho para o model.json do XGBoost.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH, help="Caminho para o metrics.json do treino.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV_PATH, help="CSV usado para recuperar metadados e opcionalmente filtrar imagens ja vistas.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Diretorio de imagens para inferencia.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob das imagens no diretorio de entrada.")
    parser.add_argument("--images", nargs="*", default=None, help="Lista explicita de imagens. Se informado, ignora o scan do diretorio.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretorio para salvar exemplos e CSV de previsoes.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold para converter probabilidade em classe. Default: usa o metrics.json.")
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade de imagens processadas.")
    parser.add_argument("--only-unseen-images", action="store_true", help="Processa apenas imagens que nao aparecem no region_dataset.csv.")
    parser.add_argument("--min-region-pixels", type=int, default=DEFAULT_MIN_REGION_PIXELS, help="Minimo de pixels por regiao na inferencia.")
    parser.add_argument("--white-threshold", type=int, default=DEFAULT_WHITE_THRESHOLD, help="Threshold para considerar pixel branco.")
    return parser.parse_args()


def load_training_metadata(metrics_path: Path) -> tuple[list[str], float, int | None]:
    if not metrics_path.exists():
        return DEFAULT_FEATURE_COLUMNS, 0.5, None

    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    threshold = float(metadata.get("parameters", {}).get("threshold", 0.5))
    best_iteration = metadata.get("model", {}).get("best_iteration")
    return feature_columns, threshold, int(best_iteration) if best_iteration is not None else None


def load_seen_image_names(dataset_csv_path: Path) -> set[str]:
    if not dataset_csv_path.exists():
        return set()

    seen_image_names: set[str] = set()
    with dataset_csv_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            seen_image_names.add(row["image_name"])
    return seen_image_names


def collect_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.images:
        image_paths = [Path(image_path) for image_path in args.images]
    else:
        image_paths = sorted(args.input_dir.glob(args.pattern), key=_numeric_image_sort_key)

    if args.only_unseen_images:
        seen_image_names = load_seen_image_names(args.dataset_csv)
        image_paths = [image_path for image_path in image_paths if image_path.name not in seen_image_names]

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    return image_paths


def _numeric_image_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def region_to_feature_row(region: RegionMetrics, feature_columns: list[str]) -> list[float]:
    min_x, min_y, max_x, max_y = region.bounds
    computed_values = {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
    }
    row: list[float] = []
    for column_name in feature_columns:
        if column_name in computed_values:
            value = computed_values[column_name]
        else:
            value = getattr(region, column_name)
        if isinstance(value, bool):
            row.append(float(int(value)))
        else:
            row.append(float(value))
    return row


def predict_regions_for_image(
    booster: xgb.Booster,
    image_processor: ImageProcessor,
    image_bytes: bytes,
    feature_columns: list[str],
    threshold: float,
    best_iteration: int | None,
    min_region_pixels: int,
    white_threshold: int,
) -> list[PredictedRegion]:
    regions = image_processor.extract_region_metrics(
        image_bytes,
        min_region_pixels=min_region_pixels,
        white_threshold=white_threshold,
    )
    if not regions:
        return []

    features = [region_to_feature_row(region, feature_columns) for region in regions]
    dmatrix = xgb.DMatrix(features, feature_names=feature_columns)
    if best_iteration is None:
        probabilities = booster.predict(dmatrix).tolist()
    else:
        probabilities = booster.predict(dmatrix, iteration_range=(0, best_iteration + 1)).tolist()

    predictions: list[PredictedRegion] = []
    for index, (region, probability) in enumerate(zip(regions, probabilities), start=1):
        predicted_label = 1 if probability >= threshold else 0
        predictions.append(
            PredictedRegion(
                region_id=index,
                probability=float(probability),
                predicted_label=predicted_label,
                bounds=region.bounds,
                metrics=region,
            )
        )
    return predictions


def paint_background_regions_black(
    image_bytes: bytes,
    predictions: list[PredictedRegion],
) -> bytes:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgba_image = image.convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Selecione um arquivo de imagem valido.") from exc

    pixels = rgba_image.load()
    width, _ = rgba_image.size
    for prediction in predictions:
        if prediction.predicted_label != 1:
            continue
        for pixel_index in prediction.metrics.pixels:
            x = pixel_index % width
            y = pixel_index // width
            pixels[x, y] = (0, 0, 0, 255)

    buffer = BytesIO()
    rgba_image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_predictions_csv(output_path: Path, rows: list[list[object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "image_name",
                "region_id",
                "probability",
                "predicted_label",
                "area",
                "min_x",
                "min_y",
                "max_x",
                "max_y",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args)
    if not image_paths:
        raise SystemExit("Nenhuma imagem encontrada para inferencia.")

    feature_columns, default_threshold, best_iteration = load_training_metadata(args.metrics)
    threshold = default_threshold if args.threshold is None else args.threshold

    booster = xgb.Booster()
    booster.load_model(args.model)
    image_processor = ImageProcessor()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[list[object]] = []
    summary: list[dict[str, object]] = []
    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        predictions = predict_regions_for_image(
            booster,
            image_processor,
            image_bytes,
            feature_columns,
            threshold,
            best_iteration,
            args.min_region_pixels,
            args.white_threshold,
        )
        output_bytes = paint_background_regions_black(image_bytes, predictions)
        output_path = args.output_dir / f"{image_path.stem}_pred.png"
        output_path.write_bytes(output_bytes)

        predicted_background_count = 0
        for prediction in predictions:
            if prediction.predicted_label == 1:
                predicted_background_count += 1
            prediction_rows.append(
                [
                    image_path.name,
                    prediction.region_id,
                    f"{prediction.probability:.8f}",
                    prediction.predicted_label,
                    prediction.metrics.area,
                    prediction.bounds[0],
                    prediction.bounds[1],
                    prediction.bounds[2],
                    prediction.bounds[3],
                ]
            )

        summary.append(
            {
                "image_name": image_path.name,
                "regions": len(predictions),
                "predicted_background_regions": predicted_background_count,
                "output": str(output_path),
            }
        )

    write_predictions_csv(args.output_dir / "predictions.csv", prediction_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
