# -*- coding: utf-8 -*-
"""
从项目根目录加载 ``.env`` 到 ``os.environ``。

依赖 ``python-dotenv``；未安装或缺少 ``.env`` 时不报错。
所有脚本与 ``recommendation_engine`` 应尽早调用 ``load_project_env()``，
确保 API 密钥仅从环境变量读取、不写死在代码中。
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
