import cv2
from pathlib import Path

import numpy as np

from model.classification_loop import (
    get_classification_model,
    render_prediction_gallery,
    visualize_results_on_frame,
)
from model.config import VIS_DIR, TEST_DATA_VIDEOS_DIR, INDICES_DIR
from model.index_bundle import IndexVariant


def _open_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    # Use mp4v for broad compatibility on Windows
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)


def _fit_to_target(frame: np.ndarray, target_size: tuple[int, int], *, pad_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Fit frame into target_size (w,h) with letterbox padding.

    OpenCV's VideoWriter expects all frames to have the exact same size. If upstream
    composition changes size slightly between frames, forcing a fixed size here avoids
    black bars/artifacts produced by the codec/backend.
    """

    th, tw = int(target_size[1]), int(target_size[0])
    h, w = frame.shape[:2]

    if (w, h) == (tw, th):
        return frame

    # Resize keeping aspect ratio, then pad to exact target.
    scale = min(tw / max(1, w), th / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    x0 = max(0, (tw - nw) // 2)
    y0 = max(0, (th - nh) // 2)
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _has_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


def _mux_audio_ffmpeg(src_video: Path, src_audio_video: Path, dst_video: Path) -> bool:
    """Copy audio stream from src_audio_video and video stream from src_video into dst_video."""
    import subprocess

    if not _has_ffmpeg():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_video),
        "-i",
        str(src_audio_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(dst_video),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def test_on_video(
    video_path: str | Path,
    *,
    index_dir: str | Path | None = None,
    index_variant: str | None = None,
    top_k: int = 3,
    max_instances: int = 3,
    show: bool = True,
    save_path: str | Path | None = None,
    keep_audio: bool = True,
    max_frames: int | None = None,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    if index_dir is None:
        identify = get_classification_model(top_k=top_k, index_variant=index_variant)
    else:
        identify = get_classification_model(top_k=top_k, index_dir=index_dir, index_variant=index_variant)

    writer: cv2.VideoWriter | None = None
    tmp_no_audio: Path | None = None
    target_size: tuple[int, int] | None = None  # (w,h)

    # Always save a video. If save_path is not provided, save under storage/vis.
    out_dir = VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        final_out = out_dir / f"annotated_{video_path.stem}.mp4"
    else:
        final_out = Path(save_path)

    def _estimate_panel_width(h: int, results_list: list[dict]) -> int:
        """Compute expected right-panel width based on top_k and the tile sizing rule."""
        # Count rows we would show (instances with predictions)
        valid = [inst for inst in results_list[:max_instances] if (inst.get("top3") or [])[:top_k]]
        n_rows = max(1, len(valid))

        # Mirror classification_loop defaults: tile_h default 308, tile_w default 220
        # tile_h = min(308, H*0.25, H/n_rows)
        max_h_per_row = max(60, int(round(h * 0.25)))
        target_h_per_row = max(60, int(h // n_rows))
        tile_h = min(308, max_h_per_row, target_h_per_row)
        tile_w = max(60, int(round(220 * (tile_h / 308.0))))
        return int(top_k) * int(tile_w)

    fixed_panel_w: int | None = None

    frame_idx = 0

    while cap.isOpened():
        if max_frames is not None and frame_idx >= int(max_frames):
            break

        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1

        # Run full pipeline (YOLO-seg -> quad -> warp -> embedding -> top-k)
        results = identify(frame)

        # Overlay detections
        overlay = visualize_results_on_frame(frame, results)

        h = overlay.shape[0]
        sep_w = 12

        # Decide a stable right-panel width (computed once, from the first processed frame)
        if fixed_panel_w is None:
            fixed_panel_w = _estimate_panel_width(h, results)

        # Gallery (one row per instance, top_k per row)
        gallery = render_prediction_gallery(
            results,
            max_instances=max_instances,
            top_k=top_k,
            video_height=h,
            max_row_fraction=0.25,
        )

        # Always allocate right panel with fixed width, even when gallery is None
        right_panel = np.zeros((h, fixed_panel_w, 3), dtype=np.uint8)
        if gallery is not None:
            gh = min(gallery.shape[0], h)
            gw = min(gallery.shape[1], fixed_panel_w)
            right_panel[0:gh, 0:gw] = gallery[0:gh, 0:gw]

        spacer = np.full((h, sep_w, 3), 40, dtype=np.uint8)
        combined = cv2.hconcat([overlay, spacer, right_panel])

        # Ensure a stable output size for BOTH preview and saving.
        if target_size is None:
            target_size = (combined.shape[1], combined.shape[0])

        combined_out = _fit_to_target(combined, target_size, pad_color=(40, 40, 40))

        if show:
            cv2.imshow("Pokemon Card Scanner - Video", combined_out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        if writer is None:
            # init writer after we know combined size
            tmp_no_audio = (final_out.parent / f"{final_out.stem}__noaudio.mp4")
            writer = _open_writer(tmp_no_audio, fps=fps, frame_size=target_size)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for: {tmp_no_audio}")

        writer.write(combined_out)

    cap.release()

    if writer is not None:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    # If we saved a no-audio file, mux audio from original video into final output.
    if final_out is not None and tmp_no_audio is not None:
        if keep_audio:
            ok = _mux_audio_ffmpeg(tmp_no_audio, video_path, final_out)
            if ok:
                try:
                    tmp_no_audio.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                # fallback: keep no-audio output
                tmp_no_audio.replace(final_out)
                print("[WARN] ffmpeg not available or mux failed; saved video without audio:", final_out)
        else:
            tmp_no_audio.replace(final_out)


if __name__ == "__main__":
    default_dir = TEST_DATA_VIDEOS_DIR
    default_video = default_dir / "test_video_1.mp4"
    # default_video = default_dir / "Should_I_Open_it_Or_Should_I_Keep_it_Sealed_-_Episode_495_-_Crown_Zenith_Elite_Trainer_Box_720P.mp4"
    # default_video = default_dir / "Opening_a_Phantasmal_Flames_Booster_Box_No_Bulk_-_Profit_or_Loss_Episode_95_720P.mp4"


    if not Path(default_video).exists():
        print(f"No default video found under: {default_dir}")
        print("Edit model/test_on_video.py and call test_on_video('path/to/video.mp4')")
    else:
        print(f"Running on: {default_video}")

        # Use migrated CLIP index dir + centered+normalized variant.
        index_dir = INDICES_DIR / "model=clip__q=high"
        test_on_video(
            default_video,
            index_dir=index_dir,
            index_variant=IndexVariant.CENTERED_NORMALIZED,
            top_k=3,
            max_instances=8,
            show=True,
        )
