#!/usr/bin/env python
# coding=utf-8

"""
捕获rich格式的控制台输出并保存为HTML
"""

import os
import sys
import subprocess
from rich.console import Console
from rich.terminal_theme import MONOKAI
import copy


def main():
    """
    运行指定的Python脚本并捕获其rich格式的输出
    """
    # 硬编码要运行的脚本和参数
    script_to_run = "examples/open_deep_research/run_gaia.py"

    # 创建输出目录
    output_dir = "output/rich_captures"
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件名
    output_name = f"{os.path.basename(script_to_run)}_test"

    # 创建可记录的控制台
    console = Console(record=True)

    try:
        # 向用户显示正在运行的命令
        command = [sys.executable, script_to_run]
        command_str = " ".join(command)
        console.print(f"运行命令: [bold cyan]{command_str}[/]")

        # 创建env字典并添加PYTHONIOENCODING
        env = copy.deepcopy(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"

        # 运行命令并实时捕获输出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # 逐行读取输出并显示
        for line in process.stdout:
            console.print(line, end="")

        # 等待进程完成
        process.wait()

        # 保存HTML输出
        html_path = f"{output_dir}/{output_name}.html"
        console.save_html(html_path, theme=MONOKAI)
        console.print(
            f"\n[bold green]控制台输出已保存到[/]: [link={html_path}]{html_path}[/link]"
        )

    except Exception as e:
        console.print(f"[bold red]发生错误[/]: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
