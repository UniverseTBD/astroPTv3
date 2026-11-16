import torch
from legacy_fixture import record_stream


def test_save_load_roundtrip(tmp_path, tiny_model, sequencer, collator):
    from transformers import AutoConfig, AutoModel

    batch = collator([sequencer.build(r) for r in record_stream(2)])
    with torch.no_grad():
        before = tiny_model(**batch)

    save_dir = tmp_path / "ckpt"
    tiny_model.save_pretrained(save_dir)

    config = AutoConfig.from_pretrained(save_dir)
    assert config.model_type == "astropt3"
    assert all(
        {"family", "source", "record_keys", "token_ids"} <= set(modality)
        for modality in config.modalities
    )
    assert [m["name"] for m in config.modalities] == [
        "images",
        "spectra",
        "Z",
        "ebv",
        "photometry",
    ]

    reloaded = AutoModel.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**batch)

    assert torch.allclose(before.loss, after.loss, atol=1e-6)
    assert torch.allclose(
        before.last_hidden_state, after.last_hidden_state, atol=1e-6
    )
