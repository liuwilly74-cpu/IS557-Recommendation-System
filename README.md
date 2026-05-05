# 留学选校推荐（IS 557）

基于历史录取 CSV 的相似度推荐与 Streamlit 前端；经历语义向量与 LLM 推荐理由依赖外部 API。

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制环境变量模板并填写密钥：

```bash
copy .env.example .env
```

编辑 `.env`，填入各服务商密钥（见下表）。**不要将 `.env` 提交到版本库。**

| 变量 | 用途 |
|------|------|
| `DASHSCOPE_API_KEY` | 推荐理由（Qwen）与用户经历实时 embedding |
| `DEEPSEEK_API_KEY` | 专业归类精修、经历标签 LLM 补全 |
| `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` | `major_process.py` 英译（可选） |
| `BAIDU_TRANSLATE_APP_ID` / `BAIDU_TRANSLATE_APP_KEY` | `Baidu_Text_transAPI.py`（可选） |

## 运行前端

中文界面：

```bash
streamlit run app.py
```

英文界面：

```bash
streamlit run app_en.py
```

默认读取项目根目录下的 `offers_processed.csv` 与向量相关 SQLite/索引文件（若存在）。

## 数据处理脚本（按需）

管线大致顺序：`raw_offers` → 翻译 → `gpa_extract` → 专业映射与标签 → `experience_embed_qwen` 等；各脚本模块顶部 docstring 中有说明。密钥均通过环境变量读取，见 `.env.example`。

## 许可证与数据

课程/作业项目用途时请遵循校方与数据来源要求。
