"""Download pinned minimal FP16 pipelines: python scripts/download_base.py all.
Use --check to verify without network. Partial files resume; each file retries once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import shutil
import requests

ROOT = Path(__file__).resolve().parents[1]
RECIPES = json.loads((ROOT / 'scripts/model_sources.json').read_text(encoding='utf-8-sig'))['models']
KEYS = {'sd15': 'sd15', 'sd21': 'sd21_768', 'sdxl': 'sdxl'}


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    if entry.get('sha256'):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest() == entry['sha256']
    if path.suffix == '.json' or path.name.endswith('.json.partial'):
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            return False
    return True


def fetch_range(entry, start, end, blocks):
    part = blocks / str(start)
    if part.exists() and part.stat().st_size == end - start + 1:
        return part
    temp = part.with_suffix('.tmp')
    url = entry['url'] + f'?download=true&offset={start}&request_time={time.time_ns()}'
    with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30, 45)) as response:
        response.raise_for_status()
        if response.status_code == 206:
            if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Unexpected response byte range')
        elif not (response.status_code == 200 and start == 0 and end == entry['bytes'] - 1):
            raise ValueError('Server did not honor requested range')
        received = 0
        with temp.open('wb') as output:
            for block in response.iter_content(64 * 1024):
                output.write(block)
                received += len(block)
                if received > end - start + 1:
                    raise ValueError('Response is larger than the requested range')
        if received != end - start + 1:
            raise ValueError('Incomplete response range')
    temp.replace(part)
    return part


def fetch(destination, entry):
    target = destination / entry['path']
    if valid(target, entry):
        return 'verified_existing'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    blocks = target.with_name(target.name + '.blocks')
    blocks.mkdir(exist_ok=True)
    for attempt in range(2):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry['bytes']:
                partial.write_bytes(b'')
                offset = 0
            # CDN may cap bandwidth per connection; bounded parallel ranges avoid
            # a single slow stream blocking a multi-gigabyte weight file.
            with ThreadPoolExecutor(max_workers=8) as pool:
                while offset < entry['bytes']:
                    size = 16 * 1024 * 1024
                    ranges = [(start, min(start + size, entry['bytes']) - 1)
                              for start in range(offset, min(offset + size * 8, entry['bytes']), size)]
                    futures = [pool.submit(fetch_range, entry, start, end, blocks) for start, end in ranges]
                    for future in futures:
                        part = future.result()
                        with partial.open('ab') as output, part.open('rb') as source:
                            shutil.copyfileobj(source, output)
                        offset += part.stat().st_size
                        part.unlink()
                    print(f'{destination.name}/{entry["path"]}: {offset / entry["bytes"]:.0%}', flush=True)
            if not valid(partial, entry):
                partial.write_bytes(b'')
                raise ValueError('Size, SHA256 or JSON validation failed')
            partial.replace(target)
            return 'downloaded_verified'
        except (requests.RequestException, OSError, ValueError) as exc:
            detail = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, requests.RequestException) else type(exc).__name__
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = f'HTTP {exc.response.status_code}'
            if attempt:
                # Do not expose signed CDN URL query parameters in logs.
                raise RuntimeError(f'{detail}: failed after 2 attempts; partial retained') from None
            print(f'{destination.name}/{entry["path"]}: interrupted ({detail}); resuming once', flush=True)


def download(model, check=False):
    recipe = RECIPES[KEYS[model]]
    destination = ROOT / 'models/base' / model
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'model': model, 'repo_id': recipe['repo_id'], 'revision': recipe['revision'],
                'variant': recipe['variant'], 'native_resolution': recipe['native_resolution'],
                'prediction_type': recipe['prediction_type'], 'total_bytes': recipe['total_bytes'],
                'complete': False, 'files': []}
    manifest_path = destination / '.download_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for entry in sorted(recipe['files'], key=lambda item: (item['path'] == 'model_index.json', item['bytes'])):
        print(f'{model}: {entry["path"]}', flush=True)
        result = {key: entry[key] for key in ('path', 'bytes', 'sha256')}
        try:
            result['status'] = ('verified' if valid(destination / entry['path'], entry) else 'missing_or_invalid') if check else fetch(destination, entry)
        except RuntimeError as exc:
            result['status'] = 'failed'
            result['error'] = str(exc)
        manifest['files'].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest['complete'] = all(item['status'] in ('verified', 'verified_existing', 'downloaded_verified') for item in manifest['files'])
    manifest['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'{model}: {"COMPLETE" if manifest["complete"] else "INCOMPLETE (see .download_manifest.json)"}', flush=True)
    return manifest


def main():
    # A second installer must not append to the same partial files concurrently.
    import msvcrt
    (ROOT / '.cache').mkdir(exist_ok=True)
    download_lock = (ROOT / '.cache/download_base.lock').open('a+b')
    download_lock.write(b'0')
    download_lock.flush()
    download_lock.seek(0)
    try:
        msvcrt.locking(download_lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print('A base-model download/check is already running. Wait for it to finish.')
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=[*KEYS, 'all'])
    parser.add_argument('--check', action='store_true', help='Verify existing files without network')
    args = parser.parse_args()
    selected = list(KEYS) if args.model == 'all' else [args.model]
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        results = list(pool.map(lambda model: download(model, args.check), selected))
    (ROOT / 'logs').mkdir(exist_ok=True)
    report = ROOT / 'logs/model_downloads.json'
    try:
        previous = json.loads(report.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = []
    combined = [item for item in previous if item.get('model') not in selected] + results
    report.write_text(json.dumps(combined, indent=2), encoding='utf-8')
    return 0 if all(result['complete'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
