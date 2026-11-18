import torch
from legacy_fixture import record_stream
from test_model import single_band_collator as single_band_collator
from test_model import single_band_config as single_band_config
from test_model import single_band_model as single_band_model
from test_model import single_band_sequencer as single_band_sequencer
from transformers import AutoConfig, AutoModel


def test_save_load_roundtrip(
    tmp_path, single_band_model, single_band_sequencer, single_band_collator
):
    batch = single_band_collator([single_band_sequencer.build(r) for r in record_stream(2)])
    with torch.no_grad():
        before = single_band_model(**batch)

    save_dir = tmp_path / "ckpt"
    single_band_model.save_pretrained(save_dir)

    config = AutoConfig.from_pretrained(save_dir)
    assert config.model_type == "astropt3"
    assert config.jetformer_gmm_k == 4
    assert all(
        {"family", "source", "record_keys", "token_ids"} <= set(modality)
        for modality in config.modalities
    )
    assert [m["name"] for m in config.modalities] == [
        "legacy_north_g",
        "spectra",
        "Z",
        "ebv",
        "photometry",
    ]
    assert config.modalities == single_band_model.config.modalities

    reloaded = AutoModel.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**batch)

    assert torch.allclose(before.loss, after.loss, atol=1e-6)
    assert torch.allclose(
        before.last_hidden_state, after.last_hidden_state, atol=1e-6
    )
