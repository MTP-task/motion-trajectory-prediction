"""Config validation and command-line parsing."""

import argparse

import pytest

from mtp.config import Config, add_config_arguments, config_from_args
from mtp.modeling.train import main


def parse(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", nargs="+", default=[Config.subject])
    add_config_arguments(parser)
    return parser.parse_args(list(argv))


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.baselines == [0, 1, 2, 3, 6, 7]
    assert cfg.result_xlsx_path.name == "baseline_results_s026_multihead.xlsx"


def test_cli_overrides_and_keeps_defaults():
    cfg = config_from_args(
        parse(
            "--pretrain-epochs",
            "5",
            "--no-use-filterbank",
            "--baselines",
            "3",
            "6",
            "--linear-lags-sec",
            "-0.2",
            "0",
        )
    )
    assert cfg.pretrain_epochs == 5
    assert cfg.use_filterbank is False
    assert cfg.baselines == [3, 6]
    assert cfg.linear_lags_sec == [-0.2, 0.0]
    assert cfg.finetune_epochs == Config().finetune_epochs


def test_subject_list_is_not_copied_into_config():
    # main() runs one Config per subject; the parsed list must not leak into Config.
    cfg = config_from_args(parse("--subject", "s008", "s009"))
    assert cfg.subject == Config().subject


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "gpu"),
        ("target_stream", "wirst2"),
        ("test_fraction", 1.5),
        ("batch_size", 0),
        ("baselines", [9]),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_subject_must_be_a_single_id():
    with pytest.raises(TypeError, match="subject"):
        Config(subject=["s008", "s009"])


def test_device_accepts_an_index():
    assert Config(device="cuda:1").device == "cuda:1"


@pytest.mark.parametrize(
    "argv", [["--device", "gpu"], ["--target-stream", "wirst2"], ["--baselines", "9"]]
)
def test_cli_rejects_invalid_values_before_loading_data(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err
