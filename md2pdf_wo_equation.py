#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import tempfile
import subprocess
import markdown
from weasyprint import HTML

def extract_mermaid_blocks(md_text):
    pattern = r'```mermaid\n(.*?)\n```'
    return re.findall(pattern, md_text, flags=re.DOTALL)

def render_mermaid_to_png(mermaid_code, output_path, scale=2, bg_color='transparent', font_family=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mmd', mode='w', encoding='utf-8') as tmp:
        tmp.write(mermaid_code)
        mmd_path = tmp.name

    cmd = ['mmdc', '-i', mmd_path, '-o', output_path,
           '--scale', str(scale),
           '--backgroundColor', bg_color]
    if font_family:
        cmd += ['--fontFamily', font_family]

    subprocess.run(cmd, check=True)

def replace_mermaid_with_img(md_text, img_paths):
    def repl(_):
        return f'<img src="file://{img_paths.pop(0)}" alt="Mermaid Diagram"/>'
    return re.sub(r'```mermaid\n(.*?)\n```', repl, md_text, flags=re.DOTALL)

def fix_image_paths(md_text, base_path):
    """
    将 Markdown 图片语法转换为 HTML <img> 标签，支持：
    - 本地图片（自动转为 file:// 绝对路径）
    - 远程图片（保留 http/https）
    - SVG 文件（本地或远程）
    """
    def repl(match):
        alt_text = match.group(1)
        img_path = match.group(2).strip()

        # 判断是否远程图片
        if img_path.startswith(('http://', 'https://')):
            src = img_path
        else:
            abs_path = os.path.abspath(os.path.join(base_path, img_path))
            src = f'file://{abs_path}'

        # 判断是否 SVG
        if img_path.lower().endswith('.svg'):
            return f'<object type="image/svg+xml" data="{src}" alt="{alt_text}" style="max-width:100%; margin:20px 0;"></object>'
        else:
            return f'<img src="{src}" alt="{alt_text}"/>'

    return re.sub(r'!\[(.*?)\]\((.*?)\)', repl, md_text)

def convert_md_to_pdf(md_path, pdf_path, scale, font_family):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 渲染 Mermaid 图
    mermaid_blocks = extract_mermaid_blocks(text)
    img_paths = []
    for idx, code in enumerate(mermaid_blocks):
        out_png = os.path.join(tempfile.gettempdir(), f'mermaid_{idx}.png')
        render_mermaid_to_png(code, out_png, scale=scale, bg_color='transparent', font_family=font_family)
        img_paths.append(out_png)

    text = replace_mermaid_with_img(text, img_paths)

    # 替换 Markdown 图片语法为 HTML 标签
    base_path = os.path.dirname(os.path.abspath(md_path))
    text = fix_image_paths(text, base_path)

    # 转为 HTML
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
    html = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
      <meta charset="utf-8"/>
      <style>
        @page {{
          @bottom-right {{
            content: counter(page);
            font-size: 10px;
            color: #666;
          }}
        }}
        body {{
          font-family: 'Noto Sans SC','Microsoft YaHei','SimHei',sans-serif;
          padding: 40px; line-height:1.6; font-size:14px; color:#333;
        }}
        h1, h2, h3, h4, h5, h6 {{
          color: #0066cc;
        }}
        img {{ max-width:100%; margin:20px 0; display:block; }}
        object {{ max-width:100%; margin:20px 0; display:block; }}
        table {{ border-collapse:collapse; width:100%; }}
        table, th, td {{ border:1px solid #ccc; padding:8px; }}
        code {{ background:#f4f4f4; padding:2px 4px; border-radius:4px; }}
        pre code {{
          display:block; padding:10px; background:#f4f4f4; overflow-x:auto;
        }}
      </style>
    </head>
    <body>{body}</body>
    </html>
    """

    HTML(string=html).write_pdf(pdf_path)
    print(f"✅ PDF saved to: {pdf_path}")

def main():
    p = argparse.ArgumentParser(description='将 Markdown（含 Mermaid、图片）转为 PDF')
    p.add_argument('--input',  '-i', required=True, help='输入的 Markdown 文件')
    p.add_argument('--output', '-o', required=True, help='输出的 PDF 文件')
    p.add_argument('--scale',  '-s', type=int, default=2, help='Mermaid 图像分辨率缩放倍数（默认2）')
    p.add_argument('--font',   '-f', default=None, help='Mermaid 渲染时使用的字体（系统需安装）')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ 找不到输入文件: {args.input}")
        return

    convert_md_to_pdf(args.input, args.output, args.scale, args.font)

if __name__ == '__main__':
    main()
