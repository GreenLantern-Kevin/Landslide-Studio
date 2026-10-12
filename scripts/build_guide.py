"""Rebuild the standalone Chinese guide after editing its Markdown source."""
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'docs/使用指南.md').read_text(encoding='utf-8')
body = markdown.markdown(source, extensions=['tables','fenced_code','toc','sane_lists'])
template = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Landslide Studio 使用指南</title>
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f9;color:#26394c;font:16px/1.9 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:36px auto;padding:45px 55px;background:white;border:1px solid #dce4eb;border-radius:12px}h1{font-size:30px;color:#153c60}h2{font-size:24px;border-top:1px solid #dce4eb;padding-top:28px;margin-top:42px}h3{font-size:19px;margin-top:28px}a{color:#2363b9;overflow-wrap:anywhere}pre{background:#132a3c;color:#e2edf7;border-radius:6px;padding:20px;overflow:auto;font:14px/1.8 Consolas,monospace}code{font-family:Consolas,monospace;overflow-wrap:anywhere}p code,li code,td code{background:#edf2f8;color:#3a628a;padding:2px 5px;border-radius:3px}table{border-collapse:collapse;width:100%;display:block;overflow:auto;margin:20px 0;font-size:14px}td,th{border:1px solid #dce4eb;padding:10px 14px;text-align:left;vertical-align:top}th{background:#eef4fb}blockquote{border-left:4px solid #638ecc;background:#f0f5fc;padding:10px 18px;margin:20px 0;color:#526d86}li{margin:7px 0}.top{display:flex;justify-content:space-between;align-items:center;gap:15px;border-bottom:1px solid #dce4eb;padding-bottom:20px;font-size:13px;color:#6b839b}.top a{border:1px solid #cbd9e7;border-radius:5px;padding:6px 12px;text-decoration:none}.toc{background:#f6f9fc;padding:18px 25px;border:1px solid #dce4eb;border-radius:6px}.toc ul{padding-left:22px}.toc li{font-size:14px;margin:3px 0}@media(max-width:700px){main{margin:0;padding:22px 18px;border:0;border-radius:0}h1{font-size:25px}body{font-size:15px}}@media print{body{background:white}main{border:0;margin:0;padding:0;max-width:none}.top,.toc{display:none}pre{white-space:pre-wrap;background:#f1f3f5;color:#24374a}h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:inherit}}
</style></head><body><main><div class="top"><span>LANDSLIDE STUDIO / 本地工作台操作手册</span><a href="http://127.0.0.1:8765">打开平台 ↗</a></div>BODY</main></body></html>'''
(ROOT/'docs/使用指南.html').write_text(template.replace('BODY',body),encoding='utf-8')
print('Built docs/使用指南.html')
