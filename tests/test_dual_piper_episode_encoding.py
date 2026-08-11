from __future__ import annotations

import h5py
import torch

from exroma_bench.recording.mobile_aloha_episode_recorder import (
    MobileAlohaEpisodeRecorder,
)


def test_dual_piper_compact_encoding_merges_grippers_and_omits_mast(tmp_path):
    joint_names = [
        "camera_stand_1_joint",
        "camera_stand_2_joint",
        *(name for index in range(1, 7) for name in (f"fl_joint{index}", f"fr_joint{index}")),
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    recorder = MobileAlohaEpisodeRecorder(
        tmp_path,
        joint_encoding=MobileAlohaEpisodeRecorder.DUAL_PIPER_COMPACT_ENCODING,
    )
    recorder._configure_joint_encoding(joint_names)

    source = torch.arange(len(joint_names), dtype=torch.float32)
    encoded = recorder._encode_joint_vector(source)

    assert recorder._recorded_joint_names == [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fl_gripper",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
        "fr_gripper",
    ]
    assert encoded.shape == (14,)
    assert torch.equal(encoded[:6], source[[2, 4, 6, 8, 10, 12]])
    assert encoded[6] == 0.5 * (source[14] - source[15])
    assert torch.equal(encoded[7:13], source[[3, 5, 7, 9, 11, 13]])
    assert encoded[13] == 0.5 * (source[16] - source[17])


def test_episode_instruction_can_be_overridden_per_episode(tmp_path):
    recorder = MobileAlohaEpisodeRecorder(
        tmp_path,
        task_instruction="default instruction",
    )
    output_path = recorder.start(
        metadata={"prompt_index": 3},
        joint_names=["joint"],
        camera_names=[],
        task_instruction="episode instruction",
    )
    recorder.finish(success=True)

    with h5py.File(output_path, "r") as episode:
        assert episode.attrs["task_instruction"] == "episode instruction"


def test_video_only_recording_removes_hdf5_after_export(tmp_path, monkeypatch):
    recorder = MobileAlohaEpisodeRecorder(tmp_path, keep_hdf5=False)
    output_path = recorder.start(
        metadata={},
        joint_names=["joint"],
        camera_names=[],
    )

    def fake_export(episode_path, *, overwrite):
        assert overwrite is True
        video_path = episode_path.with_suffix(".mp4")
        video_path.write_bytes(b"video")
        return video_path

    monkeypatch.setattr(
        "exroma_bench.recording.mobile_aloha_episode_recorder.export_three_view_video",
        fake_export,
    )
    result = recorder.finish(success=False)

    assert result == tmp_path / "episode_000000.mp4"
    assert result.is_file()
    assert not output_path.exists()


def test_robotwin_compatibility_paths_are_hard_links(tmp_path):
    recorder = MobileAlohaEpisodeRecorder(tmp_path)
    recorder.start(metadata={}, joint_names=["joint"], camera_names=[])
    assert recorder._file is not None
    recorder._file.create_dataset(
        "actions/joint_position_target",
        data=[[0.0]],
    )
    for camera_name in ("mast", "front_left", "front_right"):
        recorder._file.create_dataset(
            f"observations/images/{camera_name}/rgb_jpeg",
            data=[[1, 2, 3]],
        )

    recorder._add_robotwin_compatibility_links()

    assert recorder._file["action"].id == recorder._file[
        "actions/joint_position_target"
    ].id
    assert recorder._file["observations/images/cam_high"].id == recorder._file[
        "observations/images/mast/rgb_jpeg"
    ].id
    assert recorder._file["observations/images/cam_left_wrist"].id == recorder._file[
        "observations/images/front_left/rgb_jpeg"
    ].id
    assert recorder._file["observations/images/cam_right_wrist"].id == recorder._file[
        "observations/images/front_right/rgb_jpeg"
    ].id
    recorder.discard()
