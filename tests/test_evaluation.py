"""Synthetic tests for Phase 6 evaluation utilities."""

import configparser
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_pipeline.analytics.events import Event, EventType
from vision_pipeline.config import ModelConfig, TrackerConfig
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.evaluation import detection_eval
from vision_pipeline.evaluation.analytics_eval import (
    evaluate_line_counts,
    evaluate_zone_dwell,
    read_line_ground_truth,
)
from vision_pipeline.evaluation.detection_eval import evaluate_detector, evaluate_hota
from vision_pipeline.evaluation.error_analysis import analyze_tracking_errors
from vision_pipeline.evaluation.mot_io import (
    MOTRecord,
    iter_sequence_frames,
    load_sequence,
    mot_to_tracks_csv,
    read_mot,
    tracks_csv_to_mot,
    write_mot,
)
from vision_pipeline.evaluation.perf import benchmark_pipeline
from vision_pipeline.evaluation.report import (
    collect_environment,
    load_targets,
    save_hardware_result,
    update_baseline_markdown,
    write_phase6_report,
)
from vision_pipeline.evaluation.sweep import sweep_tracker
from vision_pipeline.evaluation.tracking_eval import (
    DetectionCache,
    evaluate_tracker_on_sequence,
    evaluate_tracking,
)
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection, FrameData


def test_mot_round_trip_filters_unmarked_and_distractor_rows(tmp_path: Path) -> None:
    """MOT GT filtering keeps marked pedestrians and conversions preserve boxes."""
    source = tmp_path / "gt.txt"
    source.write_text(
        "1,7,10,20,30,40,1,1,0.9\n1,8,15,20,20,30,0,1,0.9\n1,9,40,20,20,30,1,2,0.8\n",
        encoding="utf-8",
    )
    rows = read_mot(source, ground_truth=True)
    assert [row.track_id for row in rows] == [7]
    output = tmp_path / "tracks.csv"
    mot_output = tmp_path / "roundtrip.txt"
    mot_to_tracks_csv(source, output)
    converted = tracks_csv_to_mot(output, mot_output)
    assert converted[0].frame == 1
    assert converted[0].xyxy == (10.0, 20.0, 40.0, 60.0)
    assert read_mot(mot_output) == converted


def test_mot_malformed_row_reports_file_and_line(tmp_path: Path) -> None:
    """Malformed MOT rows fail with source location context."""
    source = tmp_path / "bad.txt"
    source.write_text("1,2,not-a-number,3,4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.txt:1"):
        read_mot(source)


def test_sequence_loader_reads_image_sequence_and_reports_missing_frame(
    tmp_path: Path,
) -> None:
    """Sequence metadata drives image loading and missing files raise errors."""
    image_dir = tmp_path / "img1"
    image_dir.mkdir(parents=True)
    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": "tiny",
        "frameRate": "10",
        "imWidth": "12",
        "imHeight": "10",
        "seqLength": "1",
        "imDir": "img1",
    }
    with (tmp_path / "seqinfo.ini").open("w", encoding="utf-8") as stream:
        config.write(stream)
    sequence = load_sequence(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing image"):
        next(iter_sequence_frames(sequence))
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "000001.jpg"), frame)
    loaded = list(iter_sequence_frames(sequence))
    assert loaded[0].frame_id == 0
    assert loaded[0].timestamp_s == 0.0
    assert loaded[0].frame.shape == frame.shape


def test_sequence_loader_requires_metadata_and_source(tmp_path: Path) -> None:
    """Invalid sequence layouts fail with explicit file/source errors."""
    with pytest.raises(FileNotFoundError, match="seqinfo.ini"):
        load_sequence(tmp_path)
    (tmp_path / "seqinfo.ini").write_text(
        "[Sequence]\nname=x\nframeRate=30\nimWidth=10\nimHeight=10\nseqLength=1\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Neither image directory"):
        load_sequence(tmp_path)


def test_sequence_loader_reads_video_and_detects_truncated_video(
    tmp_path: Path,
) -> None:
    """Video-backed sequences decode frames and reject declared-length mismatch."""
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (12, 10),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video encoding is unavailable.")
    writer.write(np.zeros((10, 12, 3), dtype=np.uint8))
    writer.release()
    (tmp_path / "seqinfo.ini").write_text(
        "[Sequence]\nname=tiny\nframeRate=10\nimWidth=12\nimHeight=10\n"
        "seqLength=2\nimDir=missing\nvideo=tiny.avi\n",
        encoding="utf-8",
    )
    sequence = load_sequence(tmp_path)
    iterator = iter_sequence_frames(sequence)
    assert next(iterator).frame_id == 0
    with pytest.raises(OSError, match="seqinfo.ini declares"):
        next(iterator)


def test_detection_and_hota_optional_adapters_report_missing_inputs(
    tmp_path: Path,
) -> None:
    """Detection eval fails clearly on absent data and HOTA skips without adapter."""
    with pytest.raises(FileNotFoundError, match="dataset YAML"):
        evaluate_detector(
            ModelConfig(weights_path=tmp_path / "weights.pt"),
            tmp_path / "data.yaml",
        )
    hota = evaluate_hota(
        sequence_directory=tmp_path / "sequence",
        predictions_path=tmp_path / "predictions.txt",
    )
    assert hota["status"] == "skipped"


def test_detection_eval_extracts_ultralytics_metrics_and_hota_adapter_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out detector validation extracts metrics; installed TrackEval skips."""
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    dataset = tmp_path / "data.yaml"
    dataset.write_text("path: data\n", encoding="utf-8")

    class FakeYolo:
        def __init__(self, path: str) -> None:
            assert path == str(weights)

        def val(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["data"] == str(dataset)
            return SimpleNamespace(
                box=SimpleNamespace(map50=0.8, map=0.6, mp=0.7, mr=0.75)
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        SimpleNamespace(YOLO=FakeYolo),
    )
    result = detection_eval.evaluate_detector(
        ModelConfig(weights_path=weights), dataset
    )
    assert result["metrics"] == {
        "map50": 0.8,
        "map50_95": 0.6,
        "precision": 0.7,
        "recall": 0.75,
    }
    monkeypatch.setattr(
        detection_eval.importlib, "import_module", lambda _name: object()
    )
    assert (
        detection_eval.evaluate_hota(
            sequence_directory=tmp_path / "sequence",
            predictions_path=tmp_path / "predictions.txt",
        )["status"]
        == "skipped"
    )


def _crossing(
    event_type: EventType,
    timestamp: float,
    line_id: str = "entrance",
) -> Event:
    return Event(event_type, timestamp, int(timestamp * 10), 1, line_id=line_id)


def test_line_count_matching_uses_time_direction_and_count_error() -> None:
    """Events outside tolerance or with the opposite direction are FP and FN."""
    truth = [
        {"timestamp_s": 2.0, "direction": "in", "line_id": "entrance"},
        {"timestamp_s": 5.0, "direction": "out", "line_id": "entrance"},
    ]
    perfect = evaluate_line_counts(
        [
            _crossing(EventType.LINE_CROSS_IN, 2.2),
            _crossing(EventType.LINE_CROSS_OUT, 4.9),
        ],
        truth,
    )
    assert perfect["f1"] == pytest.approx(1.0)
    shifted = evaluate_line_counts(
        [
            _crossing(EventType.LINE_CROSS_OUT, 2.0),
            _crossing(EventType.LINE_CROSS_OUT, 8.0),
        ],
        truth,
    )
    assert (shifted["tp"], shifted["fp"], shifted["fn"]) == (0, 2, 2)
    assert shifted["count_error_absolute"] == 0
    assert shifted["count_error_percent"] == pytest.approx(0.0)


def test_analytics_csv_errors_timeline_and_zone_unmatched(tmp_path: Path) -> None:
    """Ground-truth CSV errors are explicit and charts/unmatched visits work."""
    pytest.importorskip("matplotlib")
    labels = tmp_path / "line_ground_truth.csv"
    labels.write_text(
        "timestamp_s,direction,line_id\n3.0,IN,entrance\n",
        encoding="utf-8",
    )
    truth = read_line_ground_truth(labels)
    assert truth[0]["direction"] == "in"
    chart = tmp_path / "timeline.png"
    result = evaluate_line_counts(
        [_crossing(EventType.LINE_CROSS_IN, 3.1)],
        labels,
        timeline_path=chart,
    )
    assert result["f1"] == pytest.approx(1.0)
    assert chart.is_file()
    labels.write_text(
        "timestamp_s,direction,line_id\nbad,side,entrance\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid line ground-truth"):
        read_line_ground_truth(labels)
    dwell = evaluate_zone_dwell(
        [],
        [{"zone_id": "queue", "enter_s": 0, "exit_s": 2}],
    )
    assert dwell["unmatched_ground_truth_visits"] == 1
    assert dwell["mean_absolute_error_seconds"] is None


def test_zone_dwell_handles_zero_second_entry() -> None:
    """A zone visit beginning at time zero is paired with its exit correctly."""
    events = [
        Event(EventType.ZONE_ENTER, 0.0, 0, 4, zone_id="queue"),
        Event(EventType.ZONE_EXIT, 9.0, 9, 4, zone_id="queue"),
    ]
    result = evaluate_zone_dwell(
        events,
        [{"zone_id": "queue", "enter_s": 0, "exit_s": 10}],
    )
    assert result["matched_visits"] == 1
    assert result["mean_absolute_error_seconds"] == pytest.approx(1.0)


def test_error_analysis_counts_detector_misses_and_false_positives() -> None:
    """Frame attribution detects one missed and one extra box."""
    gt = [MOTRecord(1, 1, 0, 0, 10, 10)]
    detections = [MOTRecord(1, 100, 30, 30, 10, 10)]
    result = analyze_tracking_errors(gt, detections, [])
    assert result["detector_false_negatives"] == 1
    assert result["detector_false_positives"] == 1
    assert result["top_failure_frames"] == [{"frame": 1, "failure_count": 3}]


def test_error_analysis_validates_iou_and_top_n() -> None:
    """Failure attribution requires a valid threshold and requested count."""
    with pytest.raises(ValueError, match="iou_threshold"):
        analyze_tracking_errors([], [], [], iou_threshold=-0.1)
    with pytest.raises(ValueError, match="top_n"):
        analyze_tracking_errors([], [], [], top_n=0)


def test_error_analysis_tracks_identity_breaks_and_saves_crops(
    tmp_path: Path,
) -> None:
    """Switches, detector attribution, and annotated failure crops are emitted."""
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), np.zeros((80, 80, 3), dtype=np.uint8))
    truth = [
        MOTRecord(1, 9, 20, 20, 10, 10),
        MOTRecord(2, 9, 20, 20, 10, 10),
    ]
    predictions = [
        MOTRecord(1, 1, 20, 20, 10, 10),
        MOTRecord(2, 2, 20, 20, 10, 10),
    ]
    result = analyze_tracking_errors(
        truth,
        predictions,
        predictions,
        oracle_predictions=predictions[:1],
        image_paths={1: frame_path, 2: frame_path},
        crops_directory=tmp_path / "crops",
    )
    assert result["id_switches"] == 1
    assert result["oracle_recovered_associations"] == 0
    assert Path(result["saved_crops"][0]).is_file()
    crop = cv2.imread(result["saved_crops"][0])
    assert crop.shape[0] < 80 and crop.shape[1] < 80


class _SleepingDetector(BaseDetector):
    """Minimal duck-typed detector for measuring benchmark timing behavior."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        time.sleep(0.001)
        return []

    def warmup(self) -> None:
        """No model initialization is needed for the fake detector."""


def test_perf_excludes_warmup_and_reports_stage_and_budget_metrics() -> None:
    """Warmup samples are excluded from summaries and real-time overages count."""
    frames = [
        FrameData(np.zeros((8, 8, 3), dtype=np.uint8), index, index / 30, 30)
        for index in range(4)
    ]
    result = benchmark_pipeline(
        frames,
        _SleepingDetector(),
        ByteTracker(TrackerConfig()),
        warmup_frames=1,
        realtime_fps=1000,
        draw=False,
    )
    assert result["frames_measured"] == 3
    assert result["stage_latency_ms"]["detect"]["count"] == 3
    assert result["end_to_end"]["over_budget_frames"] == 3
    assert result["end_to_end"]["p95_ms"] > 0


def test_perf_validates_arguments_and_times_writer() -> None:
    """Invalid bounds are rejected and optional video writes are timed."""
    with pytest.raises(ValueError, match="warmup_frames"):
        benchmark_pipeline([], _SleepingDetector(), ByteTracker(), warmup_frames=-1)
    with pytest.raises(ValueError, match="realtime_fps"):
        benchmark_pipeline([], _SleepingDetector(), ByteTracker(), realtime_fps=0)

    class FakeWriter:
        def __init__(self) -> None:
            self.frames = 0

        def write(self, frame: np.ndarray) -> None:
            assert frame.shape == (8, 8, 3)
            self.frames += 1

    writer = FakeWriter()
    frames = [
        FrameData(np.zeros((8, 8, 3), dtype=np.uint8), index, index / 30, 30)
        for index in range(2)
    ]
    result = benchmark_pipeline(
        frames,
        _SleepingDetector(),
        ByteTracker(),
        warmup_frames=0,
        draw=False,
        writer=writer,
        max_duration_s=1.0,
    )
    assert writer.frames == 2
    assert result["stage_latency_ms"]["write"]["count"] == 2
    assert result["write_enabled"] is True


def test_report_contains_expected_sections_and_pending_status(tmp_path: Path) -> None:
    """The combined report renders all benchmark sections and JSON output."""
    report = write_phase6_report(
        tmp_path,
        environment={"python": "3.x", "git_commit": "abc"},
        results={
            "detection": {"status": "pending", "reason": "no labels"},
            "tracking": {"metrics": {"idf1": None}},
            "analytics": {"status": "pending"},
            "performance": {"status": "pending"},
            "sweep": {"status": "pending"},
            "error_analysis": {"known_weaknesses": ["occlusion labels unavailable"]},
        },
        targets={
            "idf1": {"metric": "tracking.metrics.idf1", "operator": ">=", "value": 0.7}
        },
    )
    content = report.read_text(encoding="utf-8")
    for section in (
        "Environment",
        "Detection",
        "Tracking",
        "Analytics",
        "Performance",
        "Sweep",
        "Error Analysis",
        "Target summary",
    ):
        assert section in content
    assert "occlusion labels unavailable" in content
    assert (tmp_path / "results.json").is_file()
    for section in (
        "detection",
        "tracking",
        "analytics",
        "performance",
        "sweep",
        "error_analysis",
    ):
        assert (tmp_path / f"{section}.json").is_file()


def test_report_environment_targets_and_baseline_helpers(tmp_path: Path) -> None:
    """Reproducibility metadata and baseline rows are saved as expected."""
    config = tmp_path / "config.yaml"
    config.write_text("enabled: true\n", encoding="utf-8")
    environment = collect_environment(
        repository_root=tmp_path,
        config_paths=[config],
    )
    assert len(environment["config_sha256"]) == 64
    result_path = save_hardware_result(
        {"performance": {"ok": True}},
        tmp_path / "results",
        {"cpu_model": "test cpu"},
    )
    assert result_path.is_file()
    baseline = tmp_path / "BASELINE.md"
    perf_result = {"end_to_end": {"fps": 25.0, "mean_ms": 40.0, "p95_ms": 45.0}}
    baseline_args = {
        "path": baseline,
        "model": "retail-n",
        "device": "cpu",
        "imgsz": 640,
        "result": perf_result,
        "environment": {
            "timestamp_utc": "2026-01-02T00:00:00Z",
            "cpu_model": "test cpu",
        },
    }
    update_baseline_markdown(**baseline_args)
    update_baseline_markdown(**baseline_args)
    assert baseline.read_text(encoding="utf-8").count("retail-n") == 2
    targets_file = tmp_path / "targets.yaml"
    targets_file.write_text(
        "quality: {metric: tracking.idf1, operator: '>=', value: 0.7}\n",
        encoding="utf-8",
    )
    assert load_targets(targets_file)["quality"]["value"] == 0.7
    invalid_targets = tmp_path / "invalid.yaml"
    invalid_targets.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_targets(invalid_targets)


def test_detection_cache_round_trip_and_weights_hash(tmp_path: Path) -> None:
    """Detection archives preserve typed boxes and invalidate on model changes."""
    cache = DetectionCache(tmp_path / "cache")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights-one")
    first = cache.cache_path(
        "tiny",
        weights_path=weights,
        imgsz=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    values = {
        0: [Detection((1, 2, 3, 4), 0.8, 0, "person")],
        1: [],
    }
    cache.save(first, values)
    assert cache.load(first) == values
    weights.write_bytes(b"weights-two")
    second = cache.cache_path(
        "tiny",
        weights_path=weights,
        imgsz=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    assert second != first
    with pytest.raises(FileNotFoundError, match="cache weights"):
        cache.cache_path(
            "tiny",
            weights_path=tmp_path / "missing.pt",
            imgsz=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )


def test_tracking_metrics_rejects_invalid_iou_threshold() -> None:
    """MOT matching thresholds are validated before optional metric work."""
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_tracking([], [], iou_threshold=1.1)


def test_tracker_sweep_validates_split_grid_and_random_trials(
    tmp_path: Path,
) -> None:
    """Sweep input validation rejects leaking, missing, and empty conditions."""
    sequence = ([MOTRecord(1, 1, 0, 0, 10, 10)], {})
    with pytest.raises(ValueError, match="disjoint"):
        sweep_tracker(
            {"train": sequence},
            train_sequences=["train"],
            validation_sequences=["train"],
            parameter_grid={"track_thresh": [0.5]},
            output_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="Unknown"):
        sweep_tracker(
            {},
            train_sequences=["train"],
            validation_sequences=["validation"],
            parameter_grid={"track_thresh": [0.5]},
            output_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="random_trials"):
        sweep_tracker(
            {"train": sequence, "validation": sequence},
            train_sequences=["train"],
            validation_sequences=["validation"],
            parameter_grid={"track_thresh": [0.5]},
            output_directory=tmp_path,
            random_trials=0,
        )


class _ConfiguredDetector(BaseDetector):
    """Count model calls while returning a deterministic person box."""

    def __init__(self, weights: Path) -> None:
        self.config = ModelConfig(weights_path=weights)
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection((2, 2, 12, 18), 0.9, 0, "person")]

    def warmup(self) -> None:
        """No inference is needed to initialize this fake detector."""


def _tiny_sequence(root: Path) -> tuple[Path, Path]:
    """Create a two-frame image sequence and MOT ground truth."""
    image_dir = root / "img1"
    image_dir.mkdir(parents=True)
    for frame_id in (1, 2):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), image)
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=tiny\nframeRate=10\nimWidth=20\nimHeight=20\n"
        "seqLength=2\nimDir=img1\n",
        encoding="utf-8",
    )
    truth_path = root / "gt.txt"
    write_mot(
        [
            MOTRecord(1, 7, 2, 2, 10, 16, 1, 1),
            MOTRecord(2, 7, 2, 2, 10, 16, 1, 1),
        ],
        truth_path,
    )
    return root, truth_path


@pytest.mark.requires_eval
def test_tracker_sequence_uses_and_reuses_detection_cache(tmp_path: Path) -> None:
    """Actual frame-wise tracker evaluation writes metrics and avoids reinference."""
    sequence_path, gt_path = _tiny_sequence(tmp_path / "sequence")
    sequence = load_sequence(sequence_path)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fake weights")
    detector = _ConfiguredDetector(weights)
    cache = DetectionCache(tmp_path / "cache")
    output = tmp_path / "predictions.txt"
    first = evaluate_tracker_on_sequence(
        detector,
        ByteTracker(TrackerConfig()),
        sequence,
        ground_truth_path=gt_path,
        predictions_path=output,
        detections_cache=cache,
    )
    assert first["metrics"]["idf1"] == pytest.approx(1.0)
    assert detector.calls == 2
    second = evaluate_tracker_on_sequence(
        detector,
        ByteTracker(TrackerConfig()),
        sequence,
        ground_truth_path=gt_path,
        detections_cache=cache,
    )
    assert second["metrics"]["idf1"] == pytest.approx(1.0)
    assert detector.calls == 2
    assert read_mot(output)[0].frame == 1


@pytest.mark.requires_eval
def test_tracker_sequence_public_oracle_and_invalid_modes(tmp_path: Path) -> None:
    """Public and oracle detections run without a detector; unknown modes fail."""
    sequence_path, gt_path = _tiny_sequence(tmp_path / "sequence")
    sequence = load_sequence(sequence_path)
    public_path = tmp_path / "public.txt"
    write_mot(read_mot(gt_path), public_path)
    for mode, public in (("public", public_path), ("oracle", None)):
        result = evaluate_tracker_on_sequence(
            None,
            ByteTracker(TrackerConfig()),
            sequence,
            ground_truth_path=gt_path,
            predictions_path=tmp_path / f"{mode}.txt",
            mode=mode,
            public_detections_path=public,
        )
        assert result["metrics"]["idf1"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="mode must"):
        evaluate_tracker_on_sequence(
            None,
            ByteTracker(),
            sequence,
            mode="unknown",
        )


@pytest.mark.requires_eval
def test_tracking_metrics_perfect_switch_and_empty_predictions() -> None:
    """MOT metrics handle perfect IDs, an ID switch, and no output tracks."""
    truth = [MOTRecord(frame, 1, 0, 0, 10, 10) for frame in (1, 2)]
    perfect = [MOTRecord(frame, 10, 0, 0, 10, 10) for frame in (1, 2)]
    assert evaluate_tracking(truth, perfect)["idf1"] == pytest.approx(1.0)
    switched = [
        MOTRecord(1, 10, 0, 0, 10, 10),
        MOTRecord(2, 11, 0, 0, 10, 10),
    ]
    assert evaluate_tracking(truth, switched)["num_switches"] == 1
    assert evaluate_tracking(truth, [])["recall"] == pytest.approx(0.0)


@pytest.mark.requires_eval
def test_tracker_sweep_selects_perfect_parameter_and_respects_split(
    tmp_path: Path,
) -> None:
    """Grid search selects the setting that retains the only synthetic person."""
    truth = [MOTRecord(1, 1, 0, 0, 10, 10)]
    detections = {
        0: [Detection((0, 0, 10, 10), 0.95, 0, "person")],
    }
    result = sweep_tracker(
        {"train": (truth, detections), "validation": (truth, detections)},
        train_sequences=["train"],
        validation_sequences=["validation"],
        parameter_grid={"min_box_area": [0.0, 1000.0]},
        output_directory=tmp_path,
    )
    assert result["best_train"]["parameters"]["min_box_area"] == 0.0
    assert result["held_out_validation"]["idf1"] == pytest.approx(1.0)
    assert result["validation_sequences"] == ["validation"]
    assert (tmp_path / "tracker_sweep.csv").is_file()
