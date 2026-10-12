from __future__ import annotations
import importlib.util
import json
import os
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
for key, relative in {'HF_HOME': '.cache/huggingface', 'TORCH_HOME': '.cache/torch', 'TMPDIR': '.cache/tmp', 'TEMP': '.cache/tmp', 'TMP': '.cache/tmp'}.items():
    folder = ROOT / relative
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

JOBS = ROOT / 'data/jobs'
JOBS.mkdir(parents=True, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
lock = threading.RLock()
heatmap_lock = threading.Lock()
jobs = {}


def now():
    return datetime.now(timezone.utc).isoformat()


from app.settings import config, system as read_system

def persist(job):
    folder = JOBS / job['id']
    folder.mkdir(exist_ok=True)
    temp = folder / 'job.tmp'
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(folder / 'job.json')


for file in JOBS.glob('*/job.json'):
    try:
        job = json.loads(file.read_text(encoding='utf-8'))
        if job['status'] in ('queued', 'running'):
            job.update(status='error', error='平台上次退出时任务被中断，请重新提交。', finished_at=now())
            persist(job)
        jobs[job['id']] = job
    except (ValueError, KeyError, OSError):
        continue

app = FastAPI(title='Landslide Studio', version='1.0.0')


@app.middleware('http')
async def fresh_frontend(request, call_next):
    response = await call_next(request)
    if request.url.path in ('/', '/index.html', '/app.js', '/styles.css'):
        response.headers['Cache-Control'] = 'no-store'
    return response


class TaskNameRequest(BaseModel):
    name: str = Field(default='', max_length=80, strict=True)

    @field_validator('name', mode='before')
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class GenerationRequest(TaskNameRequest):
    model: str = 'sd15'
    category: str = 'landslide'
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default='blurry, low quality, text, watermark, illustration', max_length=2000)
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    steps: int = Field(default=28, ge=5, le=80)
    guidance: float = Field(default=7.0, ge=1, le=20)
    lora_scale: float = Field(default=0.8, ge=0, le=2)
    seed: int = Field(default=-1, ge=-1, le=2147483643)
    count: int = Field(default=1, ge=1, le=4)


class VideoHeatmapRequest(BaseModel):
    frame_index: int = Field(ge=0, strict=True)
    speed_mode: Literal['instantaneous', 'window'] = 'instantaneous'
    qualified_only: bool = Field(default=True, strict=True)
    radius: int = Field(default=24, ge=3, le=200, strict=True)
    vmax: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    opacity: float = Field(default=0.6, ge=0.1, le=1, allow_inf_nan=False)
    plot_style: Literal['scatter', 'smooth'] = 'scatter'
    point_size: float = Field(default=12, ge=1, le=100, allow_inf_nan=False, strict=True)


def submit(kind, params, files=None):
    # Keep the display name separate from inference parameters and file paths.
    params = dict(params)
    try:
        name = TaskNameRequest(name=params.pop('name', '')).name
    except ValidationError:
        raise HTTPException(422, '任务名称必须是文字，最多80个字符。') from None
    jid = uuid.uuid4().hex
    folder = JOBS / jid
    folder.mkdir()
    job = {'id': jid, 'kind': kind, 'name': name, 'status': 'queued', 'progress': 0, 'message': '等待计算资源',
           'created_at': now(), 'params': params, 'result': None, 'error': None}
    with lock:
        jobs[jid] = job
        persist(job)
    return job


class Cancelled(Exception):
    pass


def run(jid, inputs):
    job = jobs[jid]
    folder = JOBS / jid
    def progress(percent, message):
        with lock:
            if job['status'] == 'cancelled':
                raise Cancelled()
            job.update(progress=max(0, min(99, int(percent))), message=message)
            persist(job)
    try:
        with lock:
            if job['status'] == 'cancelled':
                return
            job.update(status='running', started_at=now())
            persist(job)
        cfg = config()
        p = job['params']
        if job['kind'] == 'generation':
            from app.services.generation import run_generation
            result = run_generation(folder, p, cfg, progress)
        elif job['kind'] == 'segmentation':
            from app.services.segmentation import run_segmentation
            result = run_segmentation(inputs['rgb'], inputs.get('dem'), folder, p, cfg, progress)
        elif job['kind'] == 'video':
            from app.services.video import run_video
            result = run_video(inputs['video'], folder, p, cfg, progress)
        else:
            from app.services.analysis import run_analysis
            evidence_files = []
            if p.get('run_segmentation'):
                from app.services.segmentation import run_segmentation
                evidence_dir = folder / 'evidence'
                evidence_dir.mkdir(exist_ok=True)
                seg = run_segmentation(inputs['image'], None, evidence_dir, {}, cfg, lambda n,m: progress(n * .65,m))
                p['evidence'] = {'kind': 'segmentation', 'metrics': seg.get('metrics', {}), 'summary': seg.get('summary', '')}
                evidence_files = [{**f, 'name': 'evidence/' + f['name'], 'label': '分析依据 · ' + f['label']} for f in seg.get('files', [])]
            elif p.get('evidence_job_id'):
                evidence = jobs.get(p['evidence_job_id'])
                if not evidence or evidence['status'] != 'done' or evidence['kind'] not in ('segmentation', 'video'):
                    raise ValueError('选择的证据任务无效，请选择已完成的分割或光流任务。')
                p['evidence'] = {'kind': evidence['kind'], 'job_id': evidence['id'],
                                 'metrics': evidence['result'].get('metrics', {}),
                                 'summary': evidence['result'].get('summary', '')}
            result = run_analysis(inputs.get('image'), folder, p, cfg, progress)
            result['files'].extend(evidence_files)
        normalized = []
        for item in result.get('files', []):
            if isinstance(item, str):
                item = {'name': item, 'label': item, 'kind': 'data'}
            path = (folder / item['name']).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError('输出文件校验失败：' + item['name'])
            item['url'] = f"/api/files/{jid}/{item['name']}"
            normalized.append(item)
        result['files'] = normalized
        with lock:
            if job['status'] == 'cancelled':
                raise Cancelled()
            job.update(status='done', progress=100, message='处理完成', result=result, finished_at=now())
            persist(job)
    except Cancelled:
        with lock:
            job.update(status='cancelled', message='任务已取消', finished_at=now())
            persist(job)
    except Exception as exc:
        # Persist user-safe failures; API keys and HTTP request headers are never logged.
        message = str(exc) if isinstance(exc, (ValueError, FileNotFoundError, ImportError)) else f'{type(exc).__name__}：处理失败，请检查配置和输入文件。'
        with lock:
            if job['status'] == 'cancelled':
                job.update(message='任务已取消', finished_at=now())
            else:
                job.update(status='error', error=message, message='处理失败', finished_at=now())
            persist(job)


async def save_upload(file, folder, name, allowed):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f'不支持此文件类型：{suffix}；支持 {", ".join(sorted(allowed))}')
    target = folder / (name + suffix)
    size = 0
    try:
        with target.open('wb') as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config().get('max_upload_mb', 1024) * 1024**2:
                    raise HTTPException(413, '文件过大，请缩短视频或调整 max_upload_mb。')
                stream.write(chunk)
        if size == 0:
            raise HTTPException(400, '上传文件为空。')
        return target
    finally:
        await file.close()


IMAGES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def parse_params(raw):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except ValueError:
        raise HTTPException(400, '参数必须是 JSON 对象。')


async def upload_job(kind, p, uploads):
    job = submit(kind, p)
    try:
        inputs = {}
        for name, file, allowed in uploads:
            if file and file.filename:
                inputs[name] = await save_upload(file, JOBS/job['id'], 'input_'+name, allowed)
        executor.submit(run, job['id'], inputs)
        return job
    except Exception:
        with lock:
            jobs.pop(job['id'], None)
            shutil.rmtree(JOBS/job['id'])
        raise


@app.get('/api/system')
def system():
    return read_system()

@app.get('/api/jobs')
def list_jobs():
    with lock:
        return sorted(jobs.values(), key=lambda x: x['created_at'], reverse=True)[:200]


@app.get('/api/jobs/{jid}')
def get_job(jid: str):
    if jid not in jobs:
        raise HTTPException(404, '任务不存在。')
    return jobs[jid]


@app.post('/api/jobs/{jid}/cancel')
def cancel_job(jid: str):
    with lock:
        job = get_job(jid)
        if job['status'] not in ('queued', 'running'):
            raise HTTPException(409, '任务已结束。')
        job.update(status='cancelled', message='已请求取消，将在当前计算步骤结束后停止', finished_at=now())
        persist(job)
        return job


@app.patch('/api/jobs/{jid}/name')
def rename_job(jid: str, request: TaskNameRequest):
    with lock:
        job = get_job(jid)
        persist({**job, 'name': request.name})
        job['name'] = request.name
        return job


@app.post('/api/generation')
def generation(request: GenerationRequest):
    p = request.model_dump()
    if p['model'] not in ('sd15','sd21','sdxl') or p['category'] not in ('landslide','terrain'):
        raise HTTPException(400, '基础模型或类别无效。')
    if p['width'] % 64 or p['height'] % 64 or not p['prompt'].strip():
        raise HTTPException(400, '尺寸必须为 64 的倍数，提示词不能为空。')
    job = submit('generation', p)
    executor.submit(run, job['id'], {})
    return job


@app.post('/api/segmentation')
async def segmentation(rgb: UploadFile = File(...), dem: UploadFile | None = File(None), params: str = Form('{}')):
    p = parse_params(params)
    for name, low, high in [('input_size', 64, 1024), ('tile_overlap', 0, 1023), ('output_stride', 8, 16)]:
        if name in p and (isinstance(p[name], bool) or not isinstance(p[name], int) or not low <= p[name] <= high):
            raise HTTPException(400, f'{name} 必须为 {low}–{high} 之间的整数。')
    if 'output_stride' in p and p['output_stride'] not in (8, 16):
        raise HTTPException(400, 'output_stride 只能为 8 或 16。')
    return await upload_job('segmentation', p, [('rgb', rgb, IMAGES), ('dem', dem, IMAGES)])


@app.get('/api/segmentation/checkpoint')
def inspect_segmentation_checkpoint():
    from app.checkpoints import inspect_checkpoint, resolve_checkpoint_config
    cfg = config()['segmentation']
    path = ROOT / cfg['multimodal_weights']
    if not path.is_file():
        raise HTTPException(404, '尚未找到双模态分割权重。')
    try:
        info = inspect_checkpoint(path)
        effective, sources = resolve_checkpoint_config(cfg, info)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {'checkpoint': info, 'effective': {key: effective.get(key) for key in
            ('model_name', 'output_stride', 'num_classes', 'input_size', 'tile_overlap')}, 'sources': sources}


@app.post('/api/video')
async def video(video: UploadFile = File(...), params: str = Form('{}')):
    p = parse_params(params)
    if p.get('algorithm', 'sparse') not in ('sparse', 'dense'):
        raise HTTPException(400, '请选择稀疏光流或稠密光流。')
    return await upload_job('video', p, [('video', video, {'.mp4','.avi','.mov','.mkv','.webm'})])


@app.post('/api/video/{jid}/heatmap')
def video_heatmap(jid: str, request: VideoHeatmapRequest):
    # Read saved sparse measurements; exporting a figure never reruns optical flow.
    with lock:
        job = get_job(jid)
        result = job.get('result') or {}
        algorithm = result.get('algorithm', result.get('metrics', {}).get('algorithm'))
        if job['kind'] != 'video' or job['status'] != 'done' or algorithm != 'sparse':
            raise HTTPException(409, '请选择已完成的稀疏光流任务，再导出指定帧热力图。')
    if not heatmap_lock.acquire(blocking=False):
        raise HTTPException(409, '已有热力图正在导出，请稍后重试。')
    try:
        from app.services.video_heatmap import export_sparse_heatmap
        folder = (JOBS / jid).resolve()
        options = request.model_dump()
        frame_index = options.pop('frame_index')
        exported = export_sparse_heatmap(folder, frame_index, options)
        for item in exported.get('files', []):
            path = (folder / item['name']).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                raise RuntimeError('热力图输出文件校验失败')
            item['url'] = f"/api/files/{jid}/{item['name']}"
        with lock:
            result['files'].extend(exported['files'])
            result.setdefault('heatmap_exports', []).append(exported)
            persist(job)
        return {**exported, 'job_id': jid}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, '热力图导出失败，请确认原任务的视频、轨迹和摘要文件仍然完整。') from exc
    finally:
        heatmap_lock.release()


@app.post('/api/analysis')
async def analysis(image: UploadFile | None = File(None), params: str = Form('{}')):
    p = parse_params(params)
    # Evidence must come from a persisted task, never a browser-supplied assertion.
    p.pop('evidence', None)
    if p.get('run_segmentation') and not (image and image.filename):
        raise HTTPException(400, '请先上传要执行分割的 RGB 图像。')
    return await upload_job('analysis', p, [('image', image, IMAGES)])


@app.get('/api/files/{jid}/{name:path}')
def file(jid: str, name: str):
    job = get_job(jid)
    allowed = {f['name'] for f in (job.get('result') or {}).get('files', [])}
    if name not in allowed or job['status'] != 'done':
        raise HTTPException(404, '文件不存在或任务未完成。')
    folder = (JOBS/jid).resolve()
    target = (folder/name).resolve()
    if not target.is_relative_to(folder) or not target.is_file():
        raise HTTPException(404, '文件不存在。')
    return FileResponse(target)


@app.get('/api/guide')
def guide():
    path = ROOT/'docs/使用指南.html'
    return FileResponse(path, media_type='text/html')

app.mount('/', StaticFiles(directory=ROOT/'dist', html=True), name='frontend')
