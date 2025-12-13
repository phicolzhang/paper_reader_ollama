#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import tempfile
import subprocess
import markdown
from weasyprint import HTML
import shutil
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# pandoc / latex engine
HAS_PANDOC = shutil.which("pandoc") is not None
# LaTeX engines (optional). Note: CJK linebreaking requires extra TeX packages which may be absent.
PDF_ENGINES = ["xelatex", "lualatex", "pdflatex"]
AVAILABLE_PDF_ENGINE = next((e for e in PDF_ENGINES if shutil.which(e)), None)

def _tex_pkg_exists(pkg: str) -> bool:
    try:
        r = subprocess.run(["kpsewhich", pkg], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False

HAS_TEX_CJK_LINEBREAK = any(
    _tex_pkg_exists(p)
    for p in ["ctex.sty", "xeCJK.sty", "luatexja.sty", "CJKutf8.sty"]
)

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

def replace_mermaid_with_md_image(md_text, img_paths):
    """
    用 Markdown 图片语法替换 Mermaid block，供 pandoc/LaTeX 使用。
    """
    def repl(_):
        p = img_paths.pop(0)
        # pandoc/latex 使用普通文件路径即可（绝对路径最稳）
        return f'![]({p})'
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

def fix_image_paths_for_pandoc(md_text, base_path):
    """
    为 pandoc/LaTeX 生成 PDF：把本地图片路径改成绝对路径（不使用 file://）。
    远程图片保持 http/https 形式。
    """
    def repl(match):
        alt_text = match.group(1)
        img_path = match.group(2).strip()

        if img_path.startswith(("http://", "https://")):
            return match.group(0)

        abs_path = os.path.abspath(os.path.join(base_path, img_path))
        return f'![{alt_text}]({abs_path})'

    return re.sub(r'!\[(.*?)\]\((.*?)\)', repl, md_text)

def normalize_math_formulas_for_pandoc(md_text):
    """
    将 gpt_attention_is_all_you_need.md 中常见的非标准公式格式规范化为 pandoc 可识别的形式：
    - 多行块级公式:
        [
        ...latex...
        ]
      -> 
        $$
        ...latex...
        $$
    - 单行块级公式: [ ...latex... ] -> $$ ...latex... $$
    - 行内括号公式: (\frac{...}{...}), (d_k) 等 -> $...$（仅在内容含 \\ 或 _ 或 ^ 时转换）
    说明：为了避免误伤普通括号文本，只转换“看起来像 LaTeX”的括号内容。
    """
    lines = md_text.splitlines()
    out = []
    i = 0

    def paren_to_dollar_in_line(line: str) -> str:
        # 跳过已经在 $...$ / $$...$$ 内的内容，避免破坏 \left(...\right) 等
        res = []
        in_dollar = False
        in_ddollar = False
        j = 0

        # 保护 inline code `...`
        code_spans = []
        def protect_code(m):
            idx = len(code_spans)
            code_spans.append(m.group(0))
            return f"__CODESPAN_{idx}__"
        line2 = re.sub(r'`[^`]*`', protect_code, line)

        while j < len(line2):
            ch = line2[j]

            # handle $$ first
            if line2.startswith("$$", j):
                in_ddollar = not in_ddollar
                res.append("$$")
                j += 2
                continue
            if ch == "$" and not in_ddollar:
                in_dollar = not in_dollar
                res.append("$")
                j += 1
                continue

            if (not in_dollar) and (not in_ddollar) and ch == "(":
                # find matching ')'
                depth = 1
                k = j + 1
                while k < len(line2) and depth > 0:
                    if line2[k] == "(":
                        depth += 1
                    elif line2[k] == ")":
                        depth -= 1
                    k += 1
                if depth == 0:
                    content = line2[j + 1 : k - 1]
                    # only convert if looks like latex/math
                    if ("\\" in content) or ("_" in content) or ("^" in content):
                        res.append("$" + content + "$")
                        j = k
                        continue

            res.append(ch)
            j += 1

        line3 = "".join(res)
        # restore inline code
        for idx, s in enumerate(code_spans):
            line3 = line3.replace(f"__CODESPAN_{idx}__", s)
        return line3

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # multi-line bracket block: [ ... ] on separate lines
        if stripped == "[":
            j = i + 1
            formula_lines = []
            while j < len(lines) and lines[j].strip() != "]":
                formula_lines.append(lines[j])
                j += 1
            if j < len(lines) and lines[j].strip() == "]":
                content = "\n".join(formula_lines).strip()
                if "\\" in content:
                    out.append("$$")
                    out.append(content)
                    out.append("$$")
                    i = j + 1
                    continue

        # single-line [ ... ]
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2:
            content = stripped[1:-1].strip()
            if "\\" in content:
                out.append("$$")
                out.append(content)
                out.append("$$")
                i += 1
                continue

        out.append(paren_to_dollar_in_line(line))
        i += 1

    return "\n".join(out)

def normalize_horizontal_rules_for_pandoc(md_text: str, thickness: str = "0.4pt") -> str:
    """
    pandoc 的 LaTeX 输出会把 Markdown 分割线（---/***/___）渲染成较短的居中横线（通常 0.5\\linewidth）。
    这里将其替换为占满行宽的横线，保证与正文行宽一致。

    仅处理“单独一行”的分割线，且跳过 fenced code block。
    """
    lines = md_text.splitlines()
    out = []
    in_fence = False

    fence_re = re.compile(r"^\s*```")
    hr_re = re.compile(r"^\s*([-*_])\1\1+\s*$")  # 3+ of same char

    for line in lines:
        if fence_re.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if (not in_fence) and hr_re.match(line):
            out.append(f"\\noindent\\rule{{\\linewidth}}{{{thickness}}}")
            continue
        out.append(line)

    return "\n".join(out)

def insert_cjk_linebreak_hints_for_latex(md_text: str) -> str:
    """
    在不依赖 ctex/xeCJK/luatexja 的情况下，尽量避免 LaTeX 对中英文混排的“长行溢出截断”：
    - 在相邻 CJK 字符之间插入不可见的断行点：\\hspace{0pt}
    - 在 CJK 与 ASCII 字母/数字 边界插入 \\hspace{0pt}，让 LaTeX 有机会换行
    重要：跳过代码块/行内代码/数学公式，避免破坏内容。
    """
    protected = []

    def protect(pattern: str, text: str, flags: int = 0) -> str:
        def _repl(m):
            idx = len(protected)
            protected.append(m.group(0))
            return f"__PROTECTED_{idx}__"
        return re.sub(pattern, _repl, text, flags=flags)

    text = md_text
    # fenced code blocks
    text = protect(r"```[\s\S]*?```", text, flags=re.MULTILINE)
    # inline code
    text = protect(r"`[^`\n]*`", text)
    # math blocks $$...$$
    text = protect(r"\$\$[\s\S]*?\$\$", text, flags=re.DOTALL)
    # inline math $...$
    text = protect(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", text)

    # CJK ranges: \u4e00-\u9fff is enough for Simplified Chinese base
    cjk = r"[\u4e00-\u9fff]"
    ascii = r"[A-Za-z0-9]"

    # CJK-CJK boundary
    text = re.sub(rf"({cjk})(?={cjk})", r"\1\\hspace{0pt}", text)
    # ASCII -> CJK boundary
    text = re.sub(rf"({ascii})(?={cjk})", r"\1\\hspace{0pt}", text)
    # CJK -> ASCII boundary
    text = re.sub(rf"({cjk})(?={ascii})", r"\1\\hspace{0pt}", text)

    # restore protected parts
    for idx, s in enumerate(protected):
        text = text.replace(f"__PROTECTED_{idx}__", s)

    return text

def render_pdf_with_pandoc(md_text, base_path, pdf_path):
    """
    用 pandoc + LaTeX engine 直接生成 PDF（公式排版最接近 TeX / 你截图的效果）。
    """
    if not HAS_PANDOC or not AVAILABLE_PDF_ENGINE:
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_md:
        tmp_md.write(md_text)
        tmp_md_path = tmp_md.name

    # header: make headings blue without extra packages
    header_tex = r"""
\usepackage{xcolor}
\makeatletter
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\Large\bfseries\color{blue}}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                   {-3.25ex\@plus -1ex \@minus -.2ex}%
                                   {1.5ex \@plus .2ex}%
                                   {\normalfont\large\bfseries\color{blue}}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                   {-3.25ex\@plus -1ex \@minus -.2ex}%
                                   {1.5ex \@plus .2ex}%
                                   {\normalfont\normalsize\bfseries\color{blue}}}
\makeatother
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False, encoding="utf-8") as tmp_header:
        tmp_header.write(header_tex)
        header_path = tmp_header.name

    try:
        cmd = [
            "pandoc",
            tmp_md_path,
            "-o",
            pdf_path,
            "--pdf-engine",
            AVAILABLE_PDF_ENGINE,
            "--resource-path",
            base_path,
            "--include-in-header",
            header_path,
        ]
        # Ensure CJK glyphs exist when using xelatex/lualatex
        if AVAILABLE_PDF_ENGINE in ("xelatex", "lualatex"):
            cmd += ["-V", "mainfont=Noto Sans CJK SC"]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  警告: pandoc 生成 PDF 失败: {e}")
        return False
    finally:
        try:
            os.unlink(tmp_md_path)
            os.unlink(header_path)
        except Exception:
            pass

def render_html_with_pandoc(md_text, base_path):
    """
    pandoc 将 Markdown 转为 standalone HTML，并将公式转为 MathML（weasyprint 可直接排版，不依赖 JS）。
    """
    if not HAS_PANDOC:
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_md:
        tmp_md.write(md_text)
        tmp_md_path = tmp_md.name

    try:
        cmd = [
            "pandoc",
            tmp_md_path,
            "-f",
            "markdown",
            "-t",
            "html",
            "--standalone",
            "--mathml",
            "--resource-path",
            base_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return r.stdout
    except subprocess.CalledProcessError as e:
        print(f"⚠️  警告: pandoc 生成 HTML 失败: {e.stderr}")
        return None
    finally:
        try:
            os.unlink(tmp_md_path)
        except Exception:
            pass

def crop_image_whitespace(image_path, tolerance=5):
    """
    裁剪图像边缘的空白区域（透明或白色）
    tolerance: 颜色容差，用于判断是否为空白
    """
    if not HAS_PIL:
        return False
    
    try:
        img = Image.open(image_path)
        # 转换为 RGBA 模式以便处理透明通道
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 获取图像数据
        data = img.getdata()
        width, height = img.size
        
        # 找到非空白区域的边界
        # 检查每一行和每一列，找到第一个和最后一个非空白像素
        def is_blank(pixel):
            r, g, b, a = pixel
            # 如果是完全透明的，或者接近白色（考虑容差）
            if a < 10:  # 完全透明
                return True
            if r > 255 - tolerance and g > 255 - tolerance and b > 255 - tolerance:
                return True
            return False
        
        # 找到上边界
        top = 0
        for y in range(height):
            row_has_content = False
            for x in range(width):
                pixel = data[y * width + x]
                if not is_blank(pixel):
                    row_has_content = True
                    break
            if row_has_content:
                top = y
                break
        
        # 找到下边界
        bottom = height - 1
        for y in range(height - 1, -1, -1):
            row_has_content = False
            for x in range(width):
                pixel = data[y * width + x]
                if not is_blank(pixel):
                    row_has_content = True
                    break
            if row_has_content:
                bottom = y
                break
        
        # 找到左边界
        left = 0
        for x in range(width):
            col_has_content = False
            for y in range(height):
                pixel = data[y * width + x]
                if not is_blank(pixel):
                    col_has_content = True
                    break
            if col_has_content:
                left = x
                break
        
        # 找到右边界
        right = width - 1
        for x in range(width - 1, -1, -1):
            col_has_content = False
            for y in range(height):
                pixel = data[y * width + x]
                if not is_blank(pixel):
                    col_has_content = True
                    break
            if col_has_content:
                right = x
                break
        
        # 裁剪图像
        if left < right and top < bottom:
            cropped = img.crop((left, top, right + 1, bottom + 1))
            cropped.save(image_path, 'PNG', optimize=True)
            return True
        
        return False
    except Exception as e:
        print(f"⚠️  警告: 裁剪图像时出错: {e}")
        return False

def render_latex_to_png(latex_code, output_path, dpi=250, fontsize=24, is_block=False):
    """
    将 LaTeX 公式渲染为 PNG 图片
    """
    try:
        processed_latex = latex_code.strip()
        
        # 处理 matplotlib mathtext 不支持的命令
        # \text{} -> \mathrm{}
        def replace_text(match):
            content = match.group(1)
            if re.search(r'[\u4e00-\u9fff]', content):
                return content  # 包含中文，直接返回
            return r'\mathrm{' + content + '}'
        
        processed_latex = re.sub(r'\\text\{([^}]+)\}', replace_text, processed_latex)
        
        # 处理 \mathbb{} 命令
        processed_latex = re.sub(r'\\mathbb\{R\}', 'R', processed_latex)
        processed_latex = re.sub(r'\\mathbb\{([^}]+)\}', r'\1', processed_latex)
        
        # 处理其他不支持的符号命令
        processed_latex = re.sub(r'\\le\b', r'\\leq', processed_latex)  # \le -> \leq
        processed_latex = re.sub(r'\\ge\b', r'\\geq', processed_latex)  # \ge -> \geq
        processed_latex = re.sub(r'\\neq\b', r'\\ne', processed_latex)  # \neq -> \ne (如果 \neq 不支持)
        
        # 处理括号大小命令
        processed_latex = re.sub(r'\\bigl([\(\[\{])', lambda m: '\\left' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\bigr([\)\]\}])', lambda m: '\\right' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\Bigl([\(\[\{])', lambda m: '\\left' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\Bigr([\)\]\}])', lambda m: '\\right' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\biggl([\(\[\{])', lambda m: '\\left' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\biggr([\)\]\}])', lambda m: '\\right' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\Biggl([\(\[\{])', lambda m: '\\left' + m.group(1), processed_latex)
        processed_latex = re.sub(r'\\Biggr([\)\]\}])', lambda m: '\\right' + m.group(1), processed_latex)
        
        # 移除单独的括号大小命令
        processed_latex = re.sub(r'\\big(?!l|r|g)', '', processed_latex)
        processed_latex = re.sub(r'\\Big(?!l|r|g)', '', processed_latex)
        processed_latex = re.sub(r'\\bigg(?!l|r)', '', processed_latex)
        processed_latex = re.sub(r'\\Bigg(?!l|r)', '', processed_latex)
        processed_latex = re.sub(r'\\big[lr](?!\()', '', processed_latex)
        processed_latex = re.sub(r'\\Big[lr](?!\()', '', processed_latex)
        processed_latex = re.sub(r'\\bigg[lr](?!\()', '', processed_latex)
        processed_latex = re.sub(r'\\Bigg[lr](?!\()', '', processed_latex)
        
        # 检查是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', processed_latex):
            return False
        
        # 确保不包含 $ 符号
        processed_latex = processed_latex.replace('$', '')
        
        # 创建图形，根据公式类型调整大小
        # 系统方法：基于文本大小（14px）计算合适的渲染参数
        if is_block:
            # 块级公式：使用较大的图形和字体
            fig = plt.figure(figsize=(10, 2))
            fig.patch.set_alpha(0)
            text = fig.text(0.5, 0.5, f'${processed_latex}$', 
                          fontsize=fontsize, 
                          ha='center', va='center',
                          usetex=False)
            pad = 0.1
        else:
            # 行内公式：根据公式长度动态调整图形宽度，最小化空白
            # 估算公式宽度：每个字符约 0.5-0.7 个字体单位宽度
            # 使用紧凑的图形尺寸，只容纳公式内容
            formula_length = len(processed_latex)
            # 动态计算宽度：基础宽度 + 公式长度 * 字符宽度系数
            # 12pt 字体，每个字符约 0.15-0.2 英寸宽度
            estimated_width = max(1.2, min(6.0, formula_length * 0.1 + 0.8))
            fig = plt.figure(figsize=(estimated_width, 0.6))
            fig.patch.set_alpha(0)
            text = fig.text(0.5, 0.5, f'${processed_latex}$', 
                          fontsize=fontsize, 
                          ha='center', va='center',
                          usetex=False)
            # 使用零 padding，完全移除空白
            pad = 0.0
        
        plt.axis('off')
        # 对于行内公式，不使用 tight_layout，让 bbox_inches='tight' 自己处理裁剪
        if not is_block:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            plt.tight_layout(pad=0)
        
        # 保存为 PNG，使用 tight bbox 裁剪空白
        # 对于行内公式，pad_inches 设为 0 以完全移除空白
        if is_block:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       pad_inches=pad, transparent=True, facecolor='white')
        else:
            # 行内公式：使用更激进的裁剪，完全移除空白
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       pad_inches=0, transparent=True, facecolor='white', 
                       edgecolor='none')
            # 对行内公式进行图像后处理，精确裁剪边缘空白
            if HAS_PIL:
                crop_image_whitespace(output_path, tolerance=10)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"⚠️  警告: 无法渲染公式 '{latex_code[:50]}...': {e}")
        try:
            plt.close('all')
        except:
            pass
        return False

def extract_math_formulas(md_text):
    """
    提取所有数学公式
    返回: [(latex, is_block, original), ...]
    """
    formulas = []
    
    # 1. 提取块级公式 $$...$$
    for match in re.finditer(r'\$\$(.*?)\$\$', md_text, flags=re.DOTALL):
        latex = match.group(1).strip()
        formulas.append((latex, True, match.group(0)))
    
    # 2. 提取行内公式 $...$
    for match in re.finditer(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', md_text):
        latex = match.group(1).strip()
        formulas.append((latex, False, match.group(0)))
    
    return formulas

def replace_math_with_img(md_text, img_paths_dict):
    """
    将数学公式替换为图片标签，保持原有位置
    """
    def replace_block(match):
        latex = match.group(1).strip()
        key = (latex, True)
        if key in img_paths_dict:
            # 块级公式：独立成行，居中显示，前后保留空行
            return f'\n<img src="file://{img_paths_dict[key]}" alt="Math Formula" style="display:block; margin:20px auto; max-width:100%;"/>\n'
        return match.group(0)
    
    def replace_inline(match):
        latex = match.group(1).strip()
        key = (latex, False)
        if key in img_paths_dict:
            # 行内公式：保持在同一行，使用合适的高度与文本对齐
            # 不设置固定高度，让图片按原始尺寸显示，只限制最大高度
            return f'<img src="file://{img_paths_dict[key]}" alt="Math Formula" class="inline-math"/>'
        return match.group(0)
    
    # 先处理块级公式
    text = re.sub(r'\$\$(.*?)\$\$', replace_block, md_text, flags=re.DOTALL)
    # 再处理行内公式
    text = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', replace_inline, text)
    
    return text

def convert_math_to_images(md_text):
    """
    将 Markdown 中的数学公式转换为图片
    """
    if not HAS_MATPLOTLIB:
        return md_text, {}
    
    # 提取所有公式
    formulas = extract_math_formulas(md_text)
    
    if not formulas:
        return md_text, {}
    
    # 去重
    unique_formulas = {}
    for latex, is_block, original in formulas:
        key = (latex, is_block)
        if key not in unique_formulas:
            unique_formulas[key] = (latex, is_block)
    
    # 渲染公式为图片
    # 系统方法：基于文本大小（14px）计算合适的字体大小
    # 14px ≈ 10.5pt，行内公式使用12pt，块级公式使用 20pt
    img_paths_dict = {}
    for idx, (latex, is_block) in enumerate(unique_formulas.values()):
        out_png = os.path.join(tempfile.gettempdir(), f'math_{idx}.png')
        # 行内公式：12pt 字体，DPI 200（足够清晰且文件不会太大）
        # 块级公式：16pt 字体，DPI 200
        if render_latex_to_png(latex, out_png, dpi=200, fontsize=12 if not is_block else 20, is_block=is_block):
            img_paths_dict[(latex, is_block)] = out_png
    
    # 替换公式为图片
    text = replace_math_with_img(md_text, img_paths_dict)
    
    return text, img_paths_dict

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

    base_path = os.path.dirname(os.path.abspath(md_path))
    # 优先策略：
    # - 默认：pandoc + LaTeX 直接出 PDF（公式最稳定/最像 TeX）
    #   - 若缺少 TeX CJK 断行包，则对正文插入 \\hspace{0pt} 断行点，避免右侧截断
    if HAS_PANDOC and AVAILABLE_PDF_ENGINE:
        # Mermaid block -> markdown image (absolute path)
        text_pandoc = replace_mermaid_with_md_image(text, img_paths.copy())
        # normalize gpt-style math blocks: [ ... ] -> $$ ... $$
        text_pandoc = normalize_math_formulas_for_pandoc(text_pandoc)
        # fix image paths for pandoc
        text_pandoc = fix_image_paths_for_pandoc(text_pandoc, base_path)
        # make horizontal rules full-width
        text_pandoc = normalize_horizontal_rules_for_pandoc(text_pandoc)
        # If TeX lacks CJK linebreaking packages, add invisible break opportunities
        if not HAS_TEX_CJK_LINEBREAK:
            text_pandoc = insert_cjk_linebreak_hints_for_latex(text_pandoc)

        if render_pdf_with_pandoc(text_pandoc, base_path, pdf_path):
            print(f"✅ PDF saved to: {pdf_path}")
            return
        # pandoc 失败则回退原逻辑
        print("⚠️  警告: 已回退到 weasyprint/matplotlib 路径（公式效果可能较差且更慢）。")

    # ---- fallback: original weasyprint pipeline ----
    text = replace_mermaid_with_img(text, img_paths)

    # 替换 Markdown 图片语法为 HTML 标签
    text = fix_image_paths(text, base_path)

    # 转换数学公式为图片（matplotlib）
    text, _math_img_paths = convert_math_to_images(text)

    # 转为 HTML
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
    html = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
      <meta charset="utf-8"/>
      <style>
        body {{
          font-family: 'Noto Sans SC','Microsoft YaHei','SimHei',sans-serif;
          padding: 40px; line-height:1.6; font-size:14px; color:#333;
        }}
        h1, h2, h3, h4, h5, h6 {{
          color: #0066cc;
        }}
        img {{ max-width:100%; margin:20px 0; display:block; }}
        /* 行内数学公式样式：保持在同一行，高度与文本匹配 */
        /* 文本是 14px，行高 1.6，公式高度设为 1.3em（约 18.2px）与文本协调 */
        img.inline-math {{
          display: inline-block !important;
          vertical-align: middle !important;
          height: auto !important;
          max-height: 1.1em !important;
          width: auto !important;
          margin: 0 !important;
          padding: 0 !important;
          max-width: none !important;
        }}
        /* 块级数学公式样式：独立成行，居中显示 */
        img[alt="Math Formula"][style*="display:block"] {{
          display: block !important;
          margin: 20px auto !important;
          max-width: 90% !important;
        }}
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
    p = argparse.ArgumentParser(description='将 Markdown（含 Mermaid、图片、数学公式）转为 PDF')
    p.add_argument('--input',  '-i', required=True, help='输入的 Markdown 文件')
    p.add_argument('--output', '-o', required=True, help='输出的 PDF 文件')
    p.add_argument('--scale',  '-s', type=int, default=2, help='Mermaid 图像分辨率缩放倍数（默认2）')
    p.add_argument('--font',   '-f', default=None, help='Mermaid 渲染时使用的字体（系统需安装）')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ 找不到输入文件: {args.input}")
        return

    # 优先 pandoc；TeX 缺少 CJK 断行包时会自动改走 pandoc->HTML->weasyprint
    if not HAS_PANDOC:
        if not HAS_PANDOC:
            print("⚠️  警告: 未安装 pandoc，将回退到 weasyprint+matplotlib（更慢，公式效果较差）。")
        if not HAS_MATPLOTLIB:
            print("⚠️  警告: 未安装 matplotlib，且 pandoc/LaTeX 不可用，数学公式将无法渲染。")
            print("   请运行: pip install matplotlib")

    convert_md_to_pdf(args.input, args.output, args.scale, args.font)

if __name__ == '__main__':
    main()
