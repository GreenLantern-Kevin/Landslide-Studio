"""Local Diffusers inference. Each LoRA must match its base architecture."""
from pathlib import Path
import gc
import json
from app.model_files import base_model_ready, model_variant

TRIGGERS = {"landslide": "Southwest_China_Landslide", "terrain": "Southwest_China_Terrain"}


def compose_prompt(prompt, category):
    trigger = TRIGGERS[category]
    # Remove either known trigger before adding exactly the selected one.
    for word in TRIGGERS.values():
        prompt = prompt.replace(word, "")
    return f"{trigger}, {prompt.strip(' ,')}".rstrip(' ,')


def run_generation(out_dir: Path, params: dict, config: dict, progress):
    spec = config['generation']['models'][params['model']]
    root = Path(config['_root'])
    base = root / spec['path']
    lora = root / spec['lora_dir'] / (TRIGGERS[params['category']] + '.safetensors')
    if not base_model_ready(base):
        raise ValueError(f"基础模型未就绪：请将完整 Diffusers 模型目录放入 {base}，其中必须有 model_index.json。")
    if not lora.is_file():
        raise ValueError(f"缺少当前类别的 LoRA：{lora}。LoRA 必须与所选基础模型版本匹配。")
    try:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    except ImportError as exc:
        raise ValueError('尚未安装 AI 推理环境，请运行 02_安装AI环境.cmd 后重启平台。') from exc
    cuda = torch.cuda.is_available() and config['generation'].get('device') != 'cpu'
    device = 'cuda' if cuda else 'cpu'
    pipe = None
    try:
        progress(8, '正在读取本地基础模型与 LoRA')
        cls = StableDiffusionXLPipeline if params['model'] == 'sdxl' else StableDiffusionPipeline
        variant = model_variant(base, spec)
        load_options = {'variant': variant} if variant else {}
        pipe = cls.from_pretrained(str(base), torch_dtype=torch.float16 if cuda else torch.float32,
                                   local_files_only=True, use_safetensors=True, **load_options)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name='landslide_studio', local_files_only=True)
        pipe.set_adapters(['landslide_studio'], adapter_weights=[params['lora_scale']])
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        if cuda and config['generation'].get('cpu_offload', True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        seed = params['seed']
        if seed < 0:
            import secrets
            seed = secrets.randbelow(2**31)
        final_prompt = compose_prompt(params['prompt'], params['category'])
        outputs = []
        for index in range(params['count']):
            def callback(pipeline, step, timestep, callback_kwargs):
                progress(15 + int(80 * (index + (step + 1) / params['steps']) / params['count']),
                         f"生成第 {index+1}/{params['count']} 张 · {step+1}/{params['steps']} 步")
                return callback_kwargs
            generator = torch.Generator(device='cpu').manual_seed(seed + index)
            result = pipe(prompt=final_prompt, negative_prompt=params['negative_prompt'],
                          width=params['width'], height=params['height'], num_inference_steps=params['steps'],
                          guidance_scale=params['guidance'], generator=generator,
                          callback_on_step_end=callback)
            name = f'generated_{index+1:02d}.png'
            result.images[0].save(out_dir / name)
            outputs.append({'name': name, 'label': f'影像 {index+1}', 'kind': 'image'})
        metadata = {**params, 'effective_prompt': final_prompt, 'seed': seed, 'device': device, 'variant': variant,
                    'lora_file': lora.name, 'source': 'Stable Diffusion + matched LoRA',
                    'note': '合成影像，仅用于研究；不代表真实灾害观测。'}
        (out_dir / 'generation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        outputs.append({'name': 'generation.json', 'label': '生成参数 JSON', 'kind': 'data'})
        return {'files': outputs, 'metrics': {'seed': seed, 'count': len(outputs)-1, 'device': device},
                'summary': f"已生成 {params['count']} 张合成影像；激活词 {TRIGGERS[params['category']]}。", 'metadata': metadata}
    except torch.cuda.OutOfMemoryError as exc:
        raise ValueError('显存不足：请把尺寸降低到 512×512、数量设为 1，并关闭占用显存的软件；SDXL 可尝试 768×768。') from exc
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
