"""Finding, loading, splitting and batching the step 3 data."""

import numpy as np
import pytest
from torch.utils.data import RandomSampler, SequentialSampler

import harness
from mtp.config import Config
from mtp.dataset import (
    build_target_sequence,
    find_data_file,
    list_subjects,
    make_loader,
    make_loaders,
    prepare_dataset,
    stratified_split_indices,
)


@pytest.mark.parametrize("layout", ["flat", "subfolders"])
def test_find_data_file_in_both_layouts(tmp_path, layout):
    harness.write_mock_data(tmp_path, subject="s001", layout=layout)
    for kind in ("eeg", "trajectory", "labels"):
        assert find_data_file(tmp_path, kind, "s001", "MI").exists()


def test_missing_file_names_both_places_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        find_data_file(tmp_path, "eeg", "s001", "ME")
    assert str(tmp_path / "EEG_data" / "EEG_data_s001_ME.npy") in str(err.value)
    assert str(tmp_path / "EEG_data_s001_ME.npy") in str(err.value)


def test_list_subjects_across_layouts(tmp_path):
    harness.write_mock_data(tmp_path, subject="s002", layout="flat")
    harness.write_mock_data(tmp_path, subject="s001", layout="subfolders")
    assert list_subjects(tmp_path) == ["s001", "s002"]


def test_list_subjects_in_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_subjects(tmp_path)


def toy_trajectory(n):
    return {
        "shoulder": np.zeros((n, 150, 3)),
        "elbow": np.ones((n, 150, 3)),
        "wirst": np.full((n, 150, 3), 2.0),
        "IMU": np.full((n, 150, 6), 3.0),
    }


@pytest.mark.parametrize(
    ("stream", "dims"),
    [("wrist", 3), ("elbow", 3), ("shoulder", 3), ("imu", 6), ("all_xyz", 9), ("all_kin", 15)],
)
def test_target_streams(stream, dims):
    labels = np.array([101, 102, 103, 104] * 2)
    y, target_ids, dim_names = build_target_sequence(toy_trajectory(8), stream, labels)
    assert y.shape == (8, 150, dims)
    assert y.dtype == np.float32
    assert len(dim_names) == dims
    assert target_ids.tolist() == [0, 1, 2, 3] * 2


def test_wrist_reads_the_wirst_key():
    y, _, dim_names = build_target_sequence(
        toy_trajectory(4), "wrist", np.array([101, 102, 103, 104])
    )
    assert np.all(y == 2.0)
    assert dim_names == ["wrist_Vx", "wrist_Vy", "wrist_Vz"]


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="105"):
        build_target_sequence(toy_trajectory(2), "all_xyz", np.array([101, 105]))


def test_unknown_target_stream_is_rejected():
    with pytest.raises(ValueError, match="hand"):
        build_target_sequence(toy_trajectory(2), "hand", np.array([101, 102]))


def test_all_kin_order_and_names():
    y, _, dim_names = build_target_sequence(toy_trajectory(4), "all_kin", np.array([101] * 4))
    # shoulder (0), elbow (1), wrist (2), then the 6 IMU signals (3)
    assert y[0, 0].tolist() == [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [3.0] * 6
    assert dim_names[:3] == ["shoulder_Vx", "shoulder_Vy", "shoulder_Vz"]
    assert dim_names[6:9] == ["wrist_Vx", "wrist_Vy", "wrist_Vz"]
    assert dim_names[-1] == "IMU_wrist_rotVz"


def test_split_is_stratified_and_disjoint():
    target_ids = np.repeat(np.arange(4), 60)
    train, val, test = stratified_split_indices(target_ids, seed=42)
    assert (len(train), len(val), len(test)) == (167, 37, 36)
    assert set(train) | set(val) | set(test) == set(range(240))
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    for part in (val, test):
        counts = np.bincount(target_ids[part], minlength=4)
        assert counts.max() - counts.min() <= 1


def test_prepare_dataset_and_loader(tmp_path):
    harness.write_mock_data(tmp_path)
    cfg = Config(subject=harness.SUBJECT, data_dir=tmp_path)
    data = prepare_dataset(cfg, "ME", with_imu=True)

    assert data.train.X.shape[1:] == (6, 8, 2560)
    assert data.train.X.dtype == np.float32
    assert np.abs(data.train.X).max() <= cfg.eeg_clip_value
    assert sum(len(split.target_id) for split in data.splits()) == 48
    assert (data.n_bands, data.n_channels, data.out_dim) == (6, 8, 9)
    assert data.train.imu.shape[1:] == (150, 6)

    data.train.y_norm = data.train.y  # make_loader reads the normalized targets
    batch = next(iter(make_loader(data.train, batch_size=4, shuffle=True, with_imu=True)))
    assert [tuple(t.shape) for t in batch] == [
        (4, 6, 8, 2560),
        (4, 150, 6),
        (4, 150, 9),
        (4,),
        (4,),
    ]


def test_mi_has_no_imu_and_only_training_is_shuffled(tmp_path):
    harness.write_mock_data(tmp_path)
    data = prepare_dataset(
        Config(subject=harness.SUBJECT, data_dir=tmp_path), "MI", with_imu=False
    )
    assert all(split.imu is None for split in data.splits())

    for split in data.splits():
        split.y_norm = split.y
    loaders = make_loaders(data, batch_size=4)
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.val.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)
    assert len(next(iter(loaders.test))) == 4
