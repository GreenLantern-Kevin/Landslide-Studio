"""Targeted migration checks, using synthetic checkpoints, never research weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app.checkpoints import checkpoint_metadata, resolve_inference_config
from app.services import segmentation as seg


MODEL = 'm2f_deeplabv3plus_xception16'


class TinyDual(torch.nn.Module):
    """Capture explicit modality flags and expose deterministic logits."""
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.dem_backbone = torch.nn.Identity()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, rgb, dem=None, has_dem=None):
        assert has_dem is not None and has_dem.dtype == torch.bool
        self.calls.append((tuple(rgb.shape), bool(has_dem[0]), dem.detach().cpu().clone()))
        # Exact ties at the first column must remain background (upstream argmax).
        positive = torch.ones_like(rgb[:, 0]) * self.bias
        positive[:, :, 0] = 0
        return {'seg': torch.stack([torch.zeros_like(positive), positive], 1)}


def checkpoint(eval_resize=80):
    return {'model_state': TinyDual().state_dict(), 'schema_version': 3,
            'model_name': MODEL, 'config': {'model': MODEL, 'num_classes': 2,
            'output_stride': 16, 'separable_conv': False, 'crop_size': 256,
            'eval_resize': eval_resize}}


def test_eval_resize_is_separate_from_training_crop():
    info = checkpoint_metadata(checkpoint(320))
    cfg, sources = resolve_inference_config({'input_size': 256}, info, modern=True)
    assert cfg['input_size'] == cfg['eval_resize'] == 320
    assert info['recorded']['input_size'] == 256
    assert info['schema'] == 3 and sources['input_size'] == 'config.eval_resize'
    cfg, _ = resolve_inference_config({}, info, {'input_size': 80}, modern=True)
    assert cfg['input_size'] == 80 and cfg['eval_resize'] == 320


def test_val_size_fallback_and_no_config_fallback():
    c = checkpoint(); del c['config']['eval_resize']; c['config']['val_size'] = 96
    cfg, _ = resolve_inference_config({}, checkpoint_metadata(c), modern=True)
    assert cfg['input_size'] == 96
    c = {'model_state': TinyDual().state_dict()}
    cfg, _ = resolve_inference_config({'input_size': 512}, checkpoint_metadata(c), modern=True)
    assert cfg['input_size'] == 256


def test_checkpoint_stride_and_model_conflicts_are_rejected():
    with pytest.raises(ValueError, match='冲突'):
        resolve_inference_config({}, checkpoint_metadata(checkpoint()), {'output_stride': 8}, modern=True)
    c = checkpoint(); c['config']['model'] = 'other'
    with pytest.raises(ValueError, match='不一致'):
        checkpoint_metadata(c)


@pytest.mark.parametrize('has_dem', [False, True])
def test_strict_loading_and_auto_resize_export(tmp_path, monkeypatch, has_dem):
    model = TinyDual()
    network = SimpleNamespace(modeling=SimpleNamespace(MODEL_NAMES=(MODEL,), **{MODEL: lambda **_: model}))
    monkeypatch.setattr(seg, '_load_network', lambda _: network)
    path = tmp_path / 'new.pth'; torch.save(checkpoint(80), path)
    rgb = tmp_path / 'rgb.png'; dem = tmp_path / 'dem.png'
    Image.fromarray(np.full((53, 61, 3), 127, np.uint8)).save(rgb)
    # A real DEM with all channels zero must still use has_dem=True.
    Image.fromarray(np.zeros((53, 61, 3), np.uint8)).save(dem)
    cfg = {'_root': str(tmp_path), 'segmentation': {'backend': 'm2f_deeplab',
           'repo_path': 'vendor/M2F-DeepLab', 'multimodal_weights': str(path), 'device': 'cpu'}}
    result = seg.run_segmentation(rgb, dem if has_dem else None, tmp_path / 'out',
                                  {'alignment_confirmed': True}, cfg, lambda *_: None)
    assert model.calls[0][0] == (1, 3, 80, 80)
    assert model.calls[0][1] is has_dem
    if not has_dem:
        assert not model.calls[0][2].any()
    mask = np.array(Image.open(tmp_path / 'out/mask.png'))
    expected_grid = np.full((80, 80), 255, np.uint8); expected_grid[:, 0] = 0
    expected = np.array(Image.fromarray(expected_grid).resize((61, 53), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(mask, expected)
    assert np.load(tmp_path / 'out/probability.npy').shape == (53, 61)
    assert result['metrics']['eval_resize'] == 80
    assert result['metrics']['dem_route'] == ('real_dem' if has_dem else 'fhm')
    assert result['metrics']['reviewed_repo_commit'] == seg.REVIEWED_COMMIT
    for item in result['files']:
        assert (tmp_path / 'out' / item['name']).is_file()


def test_auto_zero_keeps_native_and_null_size_uses_config():
    m = TinyDual().eval(); rgb = np.zeros((17, 23, 3), np.uint8)
    cfg, _ = resolve_inference_config({'backend': 'm2f_deeplab'}, checkpoint_metadata(checkpoint(0)), modern=True)
    p, _, strategy = seg._predict_probability(m, rgb, None, cfg, {'input_size': None}, 'cpu', torch, lambda *_: None)
    assert strategy == 'native' and p.shape == (17, 23)


def test_modern_loading_rejects_old_extra_head(tmp_path, monkeypatch):
    network = SimpleNamespace(modeling=SimpleNamespace(MODEL_NAMES=(MODEL,), **{MODEL: lambda **_: TinyDual()}))
    monkeypatch.setattr(seg, '_load_network', lambda _: network)
    c = checkpoint(); c['model_state']['classifier.aux_classifier.0.weight'] = torch.ones(1)
    path = tmp_path / 'legacy.pth'; torch.save(c, path)
    with pytest.raises(ValueError, match='权重与模型架构不匹配'):
        seg._build_model({}, path, MODEL, tmp_path, torch)


def aux_shapes():
    return {'classifier.aux_classifier.' + k: v for k, v in {
        '0.weight': (256, 728, 3, 3), '1.weight': (256,), '1.bias': (256,),
        '1.running_mean': (256,), '1.running_var': (256,), '1.num_batches_tracked': (),
        '3.weight': (2, 256, 1, 1), '3.bias': (2,)}.items()}


def test_schema2_auxiliary_compatibility_keeps_all_inference_tensors(tmp_path, monkeypatch):
    network = SimpleNamespace(modeling=SimpleNamespace(MODEL_NAMES=(MODEL,), **{MODEL: lambda **_: TinyDual()}))
    monkeypatch.setattr(seg, '_load_network', lambda _: network)
    c = checkpoint(); c['schema_version'] = 2
    c['model_state'].update({k: torch.zeros(v) for k, v in aux_shapes().items()})
    path = tmp_path / 'trained_v2.pth'; torch.save(c, path)
    original = path.read_bytes()
    model = seg._build_model({}, path, MODEL, tmp_path, torch)
    assert model.studio_checkpoint_metadata['inference_loading']['strict'] is True
    assert len(model.studio_checkpoint_metadata['inference_loading']['omitted_training_auxiliary']) == 8
    torch.testing.assert_close(model.bias, c['model_state']['bias'])
    assert path.read_bytes() == original


@pytest.mark.parametrize('problem', ['extra', 'missing_main', 'main_shape', 'aux_shape', 'schema', 'missing_stride'])
def test_auxiliary_compatibility_never_hides_other_mismatches(problem):
    expected = {'main.weight': torch.empty((2, 3), device='meta')}
    state = {**expected, **{k: torch.empty(v, device='meta') for k, v in aux_shapes().items()}}
    metadata = {'schema': 2, 'recorded': {'model_name': MODEL, 'output_stride': 16,
                'num_classes': 2, 'separable_conv': False}}
    if problem == 'extra': state['unrecognized.weight'] = torch.empty(1, device='meta')
    if problem == 'missing_main': del state['main.weight']
    if problem == 'main_shape': state['main.weight'] = torch.empty((3, 2), device='meta')
    if problem == 'aux_shape': state['classifier.aux_classifier.0.weight'] = torch.empty((256, 1024, 3, 3), device='meta')
    if problem == 'schema': metadata['schema'] = 3
    if problem == 'missing_stride': del metadata['recorded']['output_stride']
    result, omitted = seg._inference_state(state, expected, metadata, MODEL, True)
    assert result is state and not omitted


def test_modern_rejects_non_uint8_dem_before_loading(tmp_path, monkeypatch):
    rgb = tmp_path / 'rgb.png'; dem = tmp_path / 'dem.tif'; weight = tmp_path / 'placeholder.pth'
    rgb.touch(); dem.touch(); weight.touch()
    monkeypatch.setattr(seg, '_read_image', lambda path, rgb, _: (
        np.zeros((8, 8, 3), dtype=np.uint8 if rgb else np.uint16), np.ones((8, 8), bool), None))
    cfg = {'segmentation': {'backend': 'm2f_deeplab', 'multimodal_weights': str(weight)}}
    with pytest.raises(ValueError, match='uint8 三通道'):
        seg.run_segmentation(rgb, dem, tmp_path / 'out', {}, cfg, lambda *_: None)
