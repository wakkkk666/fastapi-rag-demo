from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import json
import chromadb
import os
from fastapi import UploadFile, File
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import jieba
import uuid
from datetime import datetime
import re

app = FastAPI()

# CORS（开发阶段全开放）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型 —— 支持多PDF选择
class ChatRequest(BaseModel):
    collection_names: list[str] = []   # 改为列表，支持多选
    question: str
    mode: str = "qa"   # qa / summary

# JD岗位匹配请求模型
class JobMatchRequest(BaseModel):
    resume_collection: str   # 简历PDF的collection名
    jd_collection: str       # JD PDF的collection名

# DeepSeek客户端
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is required. Set it in .env or the environment before starting the app.")

client = OpenAI(
    api_key=deepseek_api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# CrossEncoder 重排序模型
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    "knowledge_base"
)

# ====== HTML页面 ======
@app.get("/", response_class=HTMLResponse)
def index():
    return r"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI智能招聘平台</title>
    <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;display:flex;height:100vh;background:#f1f5f9;overflow:hidden}

    /* 侧边栏 */
    .sidebar{width:220px;min-width:220px;background:#0f172a;display:flex;flex-direction:column;color:#cbd5e1;z-index:10}
    .sidebar .logo{padding:20px 20px 16px;font-size:16px;font-weight:700;color:#f8fafc;border-bottom:1px solid #1e293b;display:flex;align-items:center;gap:8px}
    .sidebar .logo span{background:linear-gradient(135deg,#0ea5e9,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .sidebar .nav-list{flex:1;padding:8px 0;overflow-y:auto}
    .sidebar .nav-item{padding:10px 20px;cursor:pointer;font-size:14px;display:flex;align-items:center;gap:10px;transition:all 0.15s;border-left:3px solid transparent;color:#94a3b8;user-select:none}
    .sidebar .nav-item:hover{background:#1e293b;color:#e2e8f0}
    .sidebar .nav-item.active{background:#1e293b;color:#f8fafc;border-left-color:#0ea5e9;font-weight:600}
    .sidebar .nav-item .nav-icon{font-size:16px;width:22px;text-align:center}

    /* 内容区 */
    .main-content{flex:1;overflow-y:auto;padding:24px}
    .section-page{display:none;animation:fadeIn 0.25s ease}
    .section-page.active{display:block}
    @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

    /* 通用卡片 */
    .card{background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);margin-bottom:16px}
    .card h3{font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px}
    .card h4{font-size:14px;font-weight:600;color:#334155;margin-bottom:8px}

    /* 统计卡片 */
    .stats-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:20px}
    .stat-card{background:#fff;border-radius:12px;padding:18px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);text-align:center;transition:transform 0.15s}
    .stat-card:hover{transform:translateY(-2px)}
    .stat-card .stat-num{font-size:28px;font-weight:800;color:#0ea5e9;line-height:1.1}
    .stat-card .stat-label{font-size:12px;color:#64748b;margin-top:4px}
    .stat-card.accent .stat-num{color:#10b981}
    .stat-card.warn .stat-num{color:#f59e0b}
    .stat-card.purple .stat-num{color:#8b5cf6}
    .stat-card.rose .stat-num{color:#f43f5e}

    /* 按钮 */
    .btn{padding:8px 18px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.15s;display:inline-flex;align-items:center;gap:6px}
    .btn:hover{opacity:0.88;transform:translateY(-1px)}
    .btn-primary{background:linear-gradient(135deg,#0ea5e9,#38bdf8);color:#fff}
    .btn-success{background:linear-gradient(135deg,#10b981,#34d399);color:#fff}
    .btn-warning{background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff}
    .btn-danger{background:linear-gradient(135deg,#ef4444,#f87171);color:#fff}
    .btn-outline{background:#fff;color:#64748b;border:1px solid #cbd5e1}
    .btn-outline:hover{background:#f8fafc}
    .btn-sm{padding:5px 12px;font-size:12px}
    .btn-lg{padding:10px 24px;font-size:15px}
    .btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}

    /* 输入框 */
    .input{border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;font-size:13px;outline:none;transition:border-color 0.15s;background:#fff}
    .input:focus{border-color:#0ea5e9;box-shadow:0 0 0 3px rgba(14,165,233,0.1)}
    .input.w-full{width:100%}
    select.input{background:#fff;cursor:pointer}

    /* 表格 */
    .table-wrap{overflow-x:auto}
    table{width:100%;border-collapse:collapse}
    table th{text-align:left;padding:10px 14px;font-size:12px;font-weight:600;color:#64748b;border-bottom:2px solid #e2e8f0;text-transform:uppercase;letter-spacing:0.5px}
    table td{padding:10px 14px;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9}
    table tbody tr:hover{background:#f8fafc}
    .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600}
    .badge-info{background:#dbeafe;color:#1e40af}
    .badge-success{background:#dcfce7;color:#166534}
    .badge-warn{background:#fef3c7;color:#92400e}

    /* 欢迎横幅 */
    .welcome-banner{background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:14px;padding:28px 32px;margin-bottom:20px;color:#f8fafc}
    .welcome-banner h1{font-size:22px;font-weight:800;margin-bottom:6px}
    .welcome-banner p{font-size:14px;color:#94a3b8}

    /* Tab 导航 */
    .tab-bar{display:flex;gap:0;border-bottom:2px solid #e2e8f0;margin-bottom:16px}
    .tab-btn{padding:10px 20px;cursor:pointer;font-size:14px;color:#64748b;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all 0.15s;background:none;font-weight:500}
    .tab-btn:hover{color:#0ea5e9}
    .tab-btn.active{color:#0ea5e9;border-bottom-color:#0ea5e9;font-weight:600}
    .tab-panel{display:none;animation:fadeIn 0.2s ease}
    .tab-panel.active{display:block}

    /* 操作行 */
    .action-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}

    /* PDF复选框 */
    .pdf-checklist{border:1px solid #e2e8f0;border-radius:8px;max-height:180px;overflow-y:auto}
    .pdf-checklist label{display:flex;align-items:center;padding:7px 14px;cursor:pointer;font-size:13px;color:#334155;transition:background 0.1s}
    .pdf-checklist label:hover{background:#f8fafc}
    .pdf-checklist input[type=checkbox]{margin-right:10px;accent-color:#0ea5e9}

    /* 空状态 */
    .empty-state{text-align:center;padding:40px 20px;color:#94a3b8}
    .empty-state .empty-icon{font-size:40px;margin-bottom:10px}

    /* 加载中 */
    .loading{text-align:center;padding:30px;color:#64748b;font-size:14px}

    /* AI回答卡片 */
    .answer-card{background:#fff;border:2px solid #0ea5e9;border-left:4px solid #0ea5e9;border-radius:12px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 8px rgba(14,165,233,0.08)}
    .answer-card .answer-label{font-size:12px;font-weight:700;color:#0ea5e9;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
    .answer-card .answer-text{font-size:15px;line-height:1.85;color:#1e293b;white-space:pre-wrap}

    /* 来源卡片 */
    .sources-header{font-size:16px;font-weight:700;color:#0f172a;margin-bottom:14px;display:flex;align-items:center;gap:8px}
    .sources-header .count-badge{background:#eff6ff;color:#0ea5e9;padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600}
    .source-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,0.03);transition:box-shadow 0.15s}
    .source-card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.06)}
    .source-card .src-meta{display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap}
    .source-card .src-meta .src-file{font-weight:700;font-size:14px;color:#0f172a}
    .source-card .src-meta .src-cid{font-size:11px;color:#94a3b8;font-family:monospace;background:#f1f5f9;padding:2px 8px;border-radius:4px}
    .source-card .src-sim{font-size:12px;padding:3px 10px;border-radius:12px;font-weight:700}
    .source-card .src-sim.high{background:#dcfce7;color:#166534}
    .source-card .src-sim.mid{background:#fef3c7;color:#92400e}
    .source-card .src-sim.low{background:#fee2e2;color:#991b1b}
    .source-card .src-preview{font-size:13px;color:#64748b;line-height:1.65;white-space:pre-wrap;margin-bottom:8px}
    .source-card .src-expanded{display:none;font-size:13px;color:#475569;line-height:1.7;white-space:pre-wrap;margin-top:10px;padding:12px;background:#f8fafc;border-radius:8px;border-left:3px solid #e2e8f0;max-height:400px;overflow-y:auto}
    .source-card .src-expanded .ctx-label{font-size:11px;font-weight:600;color:#94a3b8;margin:6px 0 2px}
    .source-card .src-expanded .ctx-cur{color:#0f172a;font-weight:500;background:#eff6ff;padding:4px 8px;border-radius:4px;display:inline-block;width:100%}
    .source-card .src-toggle{font-size:12px;font-weight:600;color:#0ea5e9;cursor:pointer;user-select:none;display:inline-block;margin-top:4px}
    .source-card .src-toggle:hover{text-decoration:underline}

    /* Markdown输出 */
    .md-output{font-size:14px;line-height:1.8;color:#334155}
    .md-output h3{font-size:17px;color:#0f172a;margin:16px 0 8px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}
    .md-output h4{font-size:15px;color:#1e293b;margin:12px 0 6px}
    .md-output strong{color:#0ea5e9}
    .md-output ul,.md-output ol{padding-left:20px;margin:6px 0}
    .md-output li{margin:3px 0}
    .md-output code{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:13px}
    .md-output hr{margin:16px 0;border:none;border-top:1px solid #e2e8f0}

    /* 设置信息区 */
    .info-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:13px}
    .info-row .info-label{color:#64748b}
    .info-row .info-value{color:#0f172a;font-weight:500}

    /* 面试占位 */
    .placeholder-card{background:#fff;border:2px dashed #cbd5e1;border-radius:16px;padding:48px 24px;text-align:center}
    .placeholder-card .ph-icon{font-size:56px;margin-bottom:16px}
    .placeholder-card .ph-title{font-size:20px;font-weight:700;color:#0f172a;margin-bottom:8px}
    .placeholder-card .ph-sub{font-size:14px;color:#94a3b8;margin-bottom:24px}
    .placeholder-card .ph-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;max-width:500px;margin:0 auto}
    .placeholder-card .ph-box{background:#f8fafc;border-radius:10px;padding:20px;text-align:center}
    .placeholder-card .ph-box .ph-box-icon{font-size:28px;margin-bottom:6px}
    .placeholder-card .ph-box .ph-box-label{font-size:13px;color:#64748b}

    /* JD卡片布局 */
    .jd-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
    .jd-card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);cursor:pointer;transition:all 0.2s;border:1px solid #e2e8f0;position:relative}
    .jd-card:hover{box-shadow:0 4px 14px rgba(0,0,0,0.08);border-color:#0ea5e9;transform:translateY(-2px)}
    .jd-card .jd-title{font-size:15px;font-weight:700;color:#0f172a;margin-bottom:2px}
    .jd-card .jd-company{font-size:12px;color:#64748b;margin-bottom:8px;display:flex;align-items:center;gap:4px}
    .jd-card .jd-tags{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
    .jd-card .jd-tag{font-size:11px;padding:2px 8px;border-radius:10px;background:#eff6ff;color:#0ea5e9;font-weight:500}
    .jd-card .jd-tag.bonus{background:#fef3c7;color:#92400e}
    .jd-card .jd-meta{display:flex;gap:14px;font-size:11px;color:#94a3b8;margin-bottom:10px}
    .jd-card .jd-meta span{display:flex;align-items:center;gap:3px}
    .jd-card .jd-actions{display:flex;gap:6px;border-top:1px solid #f1f5f9;padding-top:10px}
    .jd-card .jd-source{position:absolute;top:12px;right:12px;font-size:10px;padding:2px 6px;border-radius:8px;font-weight:500}
    .jd-card .jd-source.text{background:#dcfce7;color:#166534}
    .jd-card .jd-source.pdf{background:#dbeafe;color:#1e40af}
    .jd-card .jd-grade-badge{position:absolute;top:12px;right:52px;font-size:11px;padding:2px 8px;border-radius:8px;font-weight:700}
    .jd-card .jd-grade-badge.grade-s{background:#dcfce7;color:#166534}
    .jd-card .jd-grade-badge.grade-a{background:#dbeafe;color:#1e40af}
    .jd-card .jd-grade-badge.grade-b{background:#fef3c7;color:#92400e}
    .jd-card .jd-grade-badge.grade-c{background:#fee2e2;color:#991b1b}

    /* JD 详情弹窗 */
    .modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(15,23,42,0.5);z-index:100;justify-content:center;align-items:flex-start;padding-top:60px;overflow-y:auto}
    .modal-overlay.active{display:flex}
    .modal-box{background:#fff;border-radius:14px;width:92%;max-width:750px;max-height:85vh;overflow-y:auto;padding:28px;box-shadow:0 20px 60px rgba(0,0,0,0.2);animation:slideUp 0.25s ease;margin-bottom:40px}
    @keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
    .modal-box .modal-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}
    .modal-box .modal-header h3{font-size:18px;font-weight:700;color:#0f172a;margin:0;flex:1}
    .modal-box .modal-close{background:none;border:none;font-size:22px;color:#94a3b8;cursor:pointer;padding:0 4px;line-height:1}
    .modal-box .modal-close:hover{color:#0f172a}
    .modal-box .modal-section{margin-bottom:16px}
    .modal-box .modal-section h4{font-size:13px;font-weight:600;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px}
    .modal-box .modal-section .skill-tags{display:flex;flex-wrap:wrap;gap:6px}
    .modal-box .modal-section .skill-tags span{padding:4px 10px;border-radius:8px;font-size:12px;font-weight:500}
    .modal-box .modal-section .skill-tags .core-tag{background:#eff6ff;color:#0ea5e9}
    .modal-box .modal-section .skill-tags .bonus-tag{background:#fef3c7;color:#92400e}
    .modal-box .modal-section .focus-list{list-style:none;padding:0}
    .modal-box .modal-section .focus-list li{padding:6px 0;font-size:13px;color:#334155;border-bottom:1px solid #f1f5f9}
    .modal-box .modal-section .focus-list li:before{content:"🎯 ";margin-right:4px}
    .modal-box .jd-full-content{font-size:13px;color:#475569;line-height:1.8;white-space:pre-wrap;background:#f8fafc;padding:14px;border-radius:8px;max-height:300px;overflow-y:auto}

    /* 编辑表单 */
    .form-group{margin-bottom:10px}
    .form-group label{display:block;font-size:12px;font-weight:600;color:#64748b;margin-bottom:4px}
    .form-group .input{width:100%}
    textarea.input{resize:vertical;font-family:inherit}

    /* V2 匹配引擎样式 */
    .match-layout{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .match-score-gauge{text-align:center;padding:20px;background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:14px;color:#f8fafc}
    .match-score-gauge .big-score{font-size:56px;font-weight:900;background:linear-gradient(135deg,#0ea5e9,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .match-score-gauge .big-score.high{background:linear-gradient(135deg,#10b981,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .match-score-gauge .big-score.mid{background:linear-gradient(135deg,#f59e0b,#fbbf24);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .match-score-gauge .big-score.low{background:linear-gradient(135deg,#ef4444,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .match-score-gauge .score-label{font-size:14px;color:#94a3b8;margin-top:4px}
    .match-score-gauge .score-bars{display:flex;gap:8px;margin-top:14px}
    .match-score-gauge .score-bar-item{flex:1;text-align:center}
    .match-score-gauge .score-bar-item .bar-label{font-size:10px;color:#94a3b8;margin-bottom:3px}
    .match-score-gauge .score-bar-item .bar-track{height:6px;background:#334155;border-radius:3px;overflow:hidden}
    .match-score-gauge .score-bar-item .bar-fill{height:100%;border-radius:3px;transition:width 0.8s ease}
    .match-score-gauge .score-bar-item .bar-pct{font-size:10px;color:#94a3b8;margin-top:2px}

    .skill-coverage{margin-top:16px}
    .skill-coverage h4{font-size:13px;font-weight:600;color:#64748b;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px}
    .skill-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
    .skill-bar-row .skill-name{width:100px;font-size:12px;color:#475569;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .skill-bar-row .skill-bar-track{flex:1;height:14px;background:#f1f5f9;border-radius:7px;overflow:hidden}
    .skill-bar-row .skill-bar-fill{height:100%;border-radius:7px;transition:width 0.6s ease}
    .skill-bar-row .skill-bar-fill.hit{background:#10b981}
    .skill-bar-row .skill-bar-fill.miss{background:#fca5a5}
    .skill-bar-row .skill-status{font-size:10px;font-weight:600;width:44px;flex-shrink:0}
    .skill-bar-row .skill-status.hit{color:#166534}
    .skill-bar-row .skill-status.miss{color:#dc2626}
    .skill-bar-row .skill-tag-req{font-size:9px;padding:1px 5px;border-radius:6px;background:#dbeafe;color:#1e40af;flex-shrink:0}
    .skill-bar-row .skill-tag-pref{font-size:9px;padding:1px 5px;border-radius:6px;background:#fef3c7;color:#92400e;flex-shrink:0}

    .match-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
    .match-tags .tag-hit{font-size:11px;padding:3px 10px;border-radius:10px;background:#dcfce7;color:#166534;font-weight:500}
    .match-tags .tag-miss{font-size:11px;padding:3px 10px;border-radius:10px;background:#fee2e2;color:#991b1b;font-weight:500}
    .match-tags .tag-pref-hit{font-size:11px;padding:3px 10px;border-radius:10px;background:#fef3c7;color:#92400e;font-weight:500}

    .risk-item{padding:6px 10px;background:#fef2f2;border-left:3px solid #ef4444;border-radius:0 6px 6px 0;font-size:12px;color:#991b1b;margin-bottom:6px}

    @media(max-width:900px){.match-layout{grid-template-columns:1fr}}

    /* JD 质量分样式 */
    .jd-quality-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
    .jd-quality-label{font-size:11px;color:#94a3b8;flex-shrink:0}
    .jd-quality-bar{flex:1;height:6px;background:#f1f5f9;border-radius:3px;overflow:hidden}
    .jd-quality-fill{height:100%;border-radius:3px;transition:width 0.5s ease}
    .jd-quality-fill.high{background:#10b981}
    .jd-quality-fill.mid{background:#f59e0b}
    .jd-quality-fill.low{background:#f87171}
    .jd-quality-pct{font-size:11px;font-weight:600;flex-shrink:0}
    .jd-quality-pct.high{color:#166534}
    .jd-quality-pct.mid{color:#92400e}
    .jd-quality-pct.low{color:#991b1b}

    /* 质量徽章 */
    .quality-badge{font-size:11px;padding:3px 10px;border-radius:10px;font-weight:700;flex-shrink:0}
    .quality-badge.grade-s{background:#dcfce7;color:#166534}
    .quality-badge.grade-a{background:#dbeafe;color:#1e40af}
    .quality-badge.grade-b{background:#fef3c7;color:#92400e}
    .quality-badge.grade-c{background:#fee2e2;color:#991b1b}

    /* JD详情元数据行 */
    .jd-meta-row{display:flex;gap:24px}
    .jd-meta-item{display:flex;flex-direction:column;gap:2px}
    .jd-meta-label{font-size:11px;color:#94a3b8;font-weight:500}
    .jd-meta-val{font-size:13px;color:#0f172a;font-weight:600}

    /* 技能展开 */
    .skill-more{cursor:pointer;opacity:0.7}
    .skill-more:hover{opacity:1}

    /* ============ V4 面试界面 (SaaS风格) ============ */
    /* 顶部状态栏 */
    .iv-topbar{display:flex;align-items:center;gap:16px;background:#fff;border-radius:14px;padding:14px 20px;box-shadow:0 1px 2px rgba(0,0,0,0.03);margin-bottom:12px;flex-wrap:wrap;border:1px solid #edf0f5}
    .iv-topbar-left{display:flex;align-items:center;gap:10px;font-size:12px;flex:1;min-width:0}
    .iv-topbar-title{font-weight:700;color:#0f172a;white-space:nowrap}
    .iv-topbar-sep{color:#e2e8f0}
    .iv-topbar-sub{color:#64748b;white-space:nowrap}
    .iv-topbar-mode{color:#0ea5e9;font-weight:600;white-space:nowrap}
    .iv-topbar-progress{display:flex;flex-direction:column;gap:4px;min-width:200px}
    .iv-progress-label{display:flex;justify-content:space-between;font-size:11px;color:#94a3b8}
    .iv-progress-label b{color:#334155}
    .iv-progress-track{height:5px;background:#f1f5f9;border-radius:3px;overflow:hidden}
    .iv-progress-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#0ea5e9,#38bdf8);transition:width 0.4s ease}
    .iv-progress-eta{font-size:10px;color:#cbd5e1;text-align:right}
    .iv-topbar-actions{display:flex;gap:6px}

    /* 主布局 */
    .iv-main-layout{display:flex;gap:12px;align-items:flex-start}
    .iv-chat-area{flex:1;min-width:0}
    .iv-chat-card{border-radius:16px!important;padding:0!important;overflow:hidden}
    .iv-chat-body{min-height:58vh;max-height:64vh;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px}
    .iv-chat-footer{padding:12px 18px;border-top:1px solid #f1f5f9;display:flex;gap:8px;align-items:flex-end}
    .iv-chat-footer textarea{flex:1;border:1.5px solid #e8ecf1;border-radius:12px;padding:10px 14px;font-size:13px;resize:none;outline:none;font-family:inherit;min-height:42px;max-height:100px;line-height:1.5;background:#f8f9fb;transition:all 0.15s}
    .iv-chat-footer textarea:focus{background:#fff;border-color:#0ea5e9;box-shadow:0 0 0 3px rgba(14,165,233,0.08)}
    .iv-chat-footer button{border-radius:12px;min-width:40px;height:40px;display:flex;align-items:center;justify-content:center}

    /* 聊天气泡 */
    .iv-msg{display:flex;gap:10px;max-width:85%;animation:fadeInUp 0.25s ease}
    .iv-msg.msg-ai{align-self:flex-start}
    .iv-msg.msg-you{flex-direction:row-reverse;align-self:flex-end}
    .iv-msg .msg-avatar{width:30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0;font-weight:700}
    .iv-msg.msg-ai .msg-avatar{background:#e0e7ff;color:#4338ca}
    .iv-msg.msg-you .msg-avatar{background:#d1fae5;color:#047857}
    .iv-msg .msg-bubble{max-width:100%}
    .iv-msg .msg-tags{display:flex;gap:4px;margin-bottom:5px;flex-wrap:wrap}
    .iv-msg .msg-tag{font-size:10px;padding:2px 7px;border-radius:6px;font-weight:600;letter-spacing:0.2px;white-space:nowrap}
    .iv-msg .msg-tag.tag-deep{background:#dcfce7;color:#166534}
    .iv-msg .msg-tag.tag-tech{background:#dbeafe;color:#1e40af}
    .iv-msg .msg-tag.tag-ability{background:#fef3c7;color:#92400e}
    .iv-msg .msg-tag.tag-behavior{background:#f3e8ff;color:#6b21a8}
    .iv-msg .msg-tag.tag-followup{background:#fff7ed;color:#c2410c}
    .iv-msg .msg-tag.tag-basic{background:#f1f5f9;color:#64748b}
    .iv-msg .msg-text{padding:10px 16px;border-radius:14px;font-size:13px;line-height:1.65;word-break:break-word}
    .iv-msg.msg-ai .msg-text{background:#f4f6f9;color:#334155;border-bottom-left-radius:4px}
    .iv-msg.msg-you .msg-text{background:#e8f5e9;color:#1b4332;border-bottom-right-radius:4px}
    .iv-msg .msg-eval{margin-top:8px;padding:8px 10px;border-radius:8px;background:#fffbeb;font-size:11px;color:#92400e;line-height:1.5;border-left:2px solid #f59e0b}
    .iv-msg .msg-eval .eval-stars{font-size:13px;margin-bottom:3px}
    .iv-msg .msg-eval .eval-pros{color:#166534}
    .iv-msg .msg-eval .eval-cons{color:#c2410c;margin-top:2px}

    /* 右侧仪表盘 */
    .iv-dashboard{width:268px;flex-shrink:0;position:sticky;top:12px}
    .iv-dash-header{font-weight:700;font-size:14px;color:#0f172a;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #f1f5f9}
    .iv-dash-section{margin-bottom:14px}
    .iv-dash-section:last-child{margin-bottom:0}
    .iv-dash-section-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8;margin-bottom:6px}
    .iv-dash-perf{display:flex;align-items:center;gap:6px;font-size:13px;font-weight:600;color:#334155}
    .iv-perf-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .iv-dash-star-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:12px;color:#64748b}
    .iv-stars{font-size:13px;letter-spacing:1px}
    .iv-dash-list{list-style:none;padding:0;margin:0;font-size:11px;color:#64748b;line-height:1.6}
    .iv-dash-list li{padding:2px 0;padding-left:12px;position:relative}
    .iv-dash-list li::before{content:"•";position:absolute;left:0;color:inherit}

    /* 面试记录卡片 */
    .iv-history-card{background:#fff;border-radius:12px;padding:14px 18px;margin-bottom:8px;cursor:pointer;border:1px solid #edf0f5;transition:all 0.15s}
    .iv-history-card:hover{border-color:#0ea5e9;box-shadow:0 2px 8px rgba(14,165,233,0.08)}

    /* 面试复盘时间轴 */
    .iv-replay-timeline{max-height:70vh;overflow-y:auto;padding:4px}
    .iv-replay-item{padding:12px 0;border-bottom:1px solid #f1f5f9}
    .iv-replay-item:last-child{border-bottom:none}
    .iv-replay-round{font-size:11px;font-weight:700;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px}
    .iv-replay-bubble{padding:10px 14px;border-radius:12px;font-size:12.5px;line-height:1.6;margin:4px 0 4px 16px}
    .iv-replay-bubble.ai-bubble{background:#f4f6f9;color:#334155;border-bottom-left-radius:4px}
    .iv-replay-bubble.you-bubble{background:#e8f5e9;color:#1b4332;border-bottom-right-radius:4px}
    .iv-replay-eval{margin:4px 0 4px 16px;padding:8px 12px;background:#fffbeb;border-radius:8px;font-size:11.5px;color:#92400e;line-height:1.5;border-left:2px solid #f59e0b}

    /* ============ V3 面试模式卡片 ============ */
    .mode-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    .mode-card{padding:14px;border-radius:10px;border:2px solid #e2e8f0;cursor:pointer;transition:all 0.15s;background:#fff;text-align:center}
    .mode-card:hover{border-color:#0ea5e9;transform:translateY(-2px)}
    .mode-card.selected{border-color:#0ea5e9;background:#eff6ff;box-shadow:0 0 0 3px rgba(14,165,233,0.15)}
    .mode-card .mode-icon{font-size:28px;margin-bottom:6px}
    .mode-card .mode-name{font-size:13px;font-weight:700;color:#0f172a;margin-bottom:4px}
    .mode-card .mode-desc{font-size:10px;color:#94a3b8;line-height:1.4;margin-bottom:8px}
    .mode-card .mode-tags{display:flex;flex-wrap:wrap;gap:4px;justify-content:center}
    .mode-card .mode-tags span{font-size:9px;padding:1px 6px;border-radius:6px;background:#f1f5f9;color:#64748b}

    /* V3 面试状态面板 */
    .iv-stat-item{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #f1f5f9}
    .iv-stat-item:last-child{border-bottom:none}
    .iv-stat-label{font-size:11px;color:#94a3b8}
    .iv-stat-val{font-size:11px;font-weight:700;color:#334155}
    .iv-stat-val.mode-intern{color:#10b981}
    .iv-stat-val.mode-standard{color:#0ea5e9}
    .iv-stat-val.mode-bigtech{color:#f59e0b}
    .iv-stat-val.mode-pressure{color:#ef4444}

    /* 报告参数栏 */
    .report-param-bar{display:flex;align-items:center;gap:6px;margin-bottom:4px}
    .report-param-bar .param-name{font-size:10px;color:#64748b;width:50px;text-align:right;flex-shrink:0}
    .report-param-bar .param-track{flex:1;height:6px;background:#f1f5f9;border-radius:3px;overflow:hidden}
    .report-param-bar .param-fill{height:100%;border-radius:3px}
    .report-param-bar .param-val{font-size:10px;font-weight:700;color:#334155;width:32px;flex-shrink:0}
    @media(max-width:900px){.iv-main-layout{flex-direction:column}.iv-dashboard{width:100%;position:static}.iv-dash-section{margin-bottom:8px}}
    @media(max-width:900px){.mode-grid{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:900px){.interview-layout{grid-template-columns:1fr}}    .interview-score-panel{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.04);padding:16px}
    .interview-score-panel h4{font-size:13px;font-weight:600;color:#0f172a;margin-bottom:12px}
    .interview-score-panel .score-dim{margin-bottom:10px}
    .interview-score-panel .score-dim .dim-label{font-size:11px;color:#64748b;display:flex;justify-content:space-between;margin-bottom:3px}
    .interview-score-panel .score-dim .dim-bar{height:6px;background:#f1f5f9;border-radius:3px;overflow:hidden}
    .interview-score-panel .score-dim .dim-fill{height:100%;border-radius:3px;transition:width 0.5s}
    .interview-score-panel .overall-score{text-align:center;padding:8px;background:#f8fafc;border-radius:8px;margin-bottom:10px}
    .interview-score-panel .overall-score .big-num{font-size:36px;font-weight:800;color:#0ea5e9}

    @media(max-width:900px){.interview-layout{grid-template-columns:1fr}}

    /* ============ 统一资料中心 V2 样式 ============ */
    .doc-filter-bar{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
    .doc-filter-btn{padding:6px 16px;border-radius:20px;font-size:12px;font-weight:500;cursor:pointer;border:1px solid #e2e8f0;background:#fff;color:#64748b;transition:all 0.15s;white-space:nowrap}
    .doc-filter-btn:hover{border-color:#0ea5e9;color:#0ea5e9}
    .doc-filter-btn.active{background:linear-gradient(135deg,#0ea5e9,#38bdf8);color:#fff;border-color:transparent}
    .doc-filter-btn .filter-count{font-size:10px;margin-left:4px;opacity:0.75}

    .doc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
    .doc-card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);cursor:pointer;transition:all 0.2s;border:1px solid #e2e8f0;position:relative;overflow:hidden}
    .doc-card:hover{box-shadow:0 4px 14px rgba(0,0,0,0.08);border-color:#0ea5e9;transform:translateY(-2px)}
    .doc-card .doc-icon{font-size:32px;margin-bottom:8px}
    .doc-card .doc-filename{font-size:14px;font-weight:700;color:#0f172a;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .doc-card .doc-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
    .doc-card .doc-meta span{font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:3px}
    .doc-card .doc-tags{display:flex;gap:6px;margin-bottom:10px}
    .doc-card .doc-tag{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600;text-transform:uppercase}
    .doc-card .doc-tag.type-jd{background:#dbeafe;color:#1e40af}
    .doc-card .doc-tag.type-resume{background:#dcfce7;color:#166534}
    .doc-card .doc-tag.type-knowledge{background:#fef3c7;color:#92400e}
    .doc-card .doc-source-tag{font-size:10px;padding:2px 8px;border-radius:8px;font-weight:500}
    .doc-card .doc-source-tag.src-pdf{background:#fff1f2;color:#be123c}
    .doc-card .doc-source-tag.src-image{background:#f0f9ff;color:#0369a1}
    .doc-card .doc-source-tag.src-text{background:#f5f3ff;color:#6d28d9}
    .doc-card .doc-actions{display:flex;gap:6px;border-top:1px solid #f1f5f9;padding-top:10px}
    .doc-card .doc-bar{position:absolute;top:0;left:0;height:3px;width:100%}
    .doc-card .doc-bar.bar-jd{background:linear-gradient(90deg,#3b82f6,#60a5fa)}
    .doc-card .doc-bar.bar-resume{background:linear-gradient(90deg,#10b981,#34d399)}
    .doc-card .doc-bar.bar-knowledge{background:linear-gradient(90deg,#f59e0b,#fbbf24)}

    /* 上传模态窗 */
    .upload-modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(15,23,42,0.5);z-index:100;justify-content:center;align-items:flex-start;padding-top:80px}
    .upload-modal-overlay.active{display:flex}
    .upload-modal{background:#fff;border-radius:14px;width:92%;max-width:500px;max-height:85vh;overflow-y:auto;padding:28px;box-shadow:0 20px 60px rgba(0,0,0,0.2);animation:slideUp 0.25s ease}
    .upload-modal h3{font-size:18px;font-weight:700;color:#0f172a;margin-bottom:16px}
    .upload-modal .type-selector{display:flex;gap:10px;margin-bottom:16px}
    .upload-modal .type-option{flex:1;text-align:center;padding:16px 12px;border-radius:10px;border:2px solid #e2e8f0;cursor:pointer;transition:all 0.15s;background:#fff}
    .upload-modal .type-option:hover{border-color:#0ea5e9}
    .upload-modal .type-option.selected{border-color:#0ea5e9;background:#eff6ff}
    .upload-modal .type-option .type-icon{font-size:24px;margin-bottom:6px}
    .upload-modal .type-option .type-label{font-size:12px;font-weight:600;color:#334155}
    .upload-modal .type-option .type-desc{font-size:10px;color:#94a3b8;margin-top:2px}
    .upload-modal .file-drop{width:100%;padding:32px 16px;border:2px dashed #cbd5e1;border-radius:10px;text-align:center;cursor:pointer;transition:all 0.15s;margin-bottom:12px}
    .upload-modal .file-drop:hover{border-color:#0ea5e9;background:#f8fafc}
    .upload-modal .file-drop.has-file{border-style:solid;border-color:#10b981;background:#f0fdf4}
    .upload-modal .file-drop .drop-icon{font-size:36px;margin-bottom:8px}
    .upload-modal .file-drop .drop-text{font-size:13px;color:#64748b}
    .upload-modal .file-drop .drop-hint{font-size:11px;color:#94a3b8;margin-top:4px}
    .upload-modal .file-drop .file-name{font-size:13px;color:#0f172a;font-weight:600;margin-top:6px}
    .upload-modal .upload-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}

    /* Chunk列表 */
    .chunk-list{max-height:400px;overflow-y:auto}
    .chunk-item{background:#f8fafc;border-radius:8px;padding:10px 14px;margin-bottom:8px;border:1px solid #e2e8f0}
    .chunk-item .chunk-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
    .chunk-item .chunk-num{font-size:11px;font-weight:600;color:#64748b}
    .chunk-item .chunk-len{font-size:10px;color:#94a3b8}
    .chunk-item .chunk-preview{font-size:12px;color:#475569;line-height:1.6;white-space:pre-wrap}
    .chunk-item .chunk-full{display:none;font-size:12px;color:#334155;line-height:1.7;white-space:pre-wrap;margin-top:10px;padding:10px;background:#fff;border-radius:6px;border:1px solid #e2e8f0;max-height:300px;overflow-y:auto}
    .chunk-item .chunk-toggle{font-size:11px;color:#0ea5e9;cursor:pointer;font-weight:600;margin-top:4px;display:inline-block}
    .chunk-item .chunk-toggle:hover{text-decoration:underline}

    /* 空状态上传按钮 */
    .empty-upload-btn{display:inline-flex;align-items:center;gap:6px;margin-top:12px;padding:8px 20px;background:linear-gradient(135deg,#0ea5e9,#38bdf8);color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.15s}
    .empty-upload-btn:hover{opacity:0.9;transform:translateY(-1px)}
    </style>
    </head>
    <body>

    <!-- 侧边栏 -->
    <nav class="sidebar">
        <div class="logo"><span>AI</span> 智能招聘平台</div>
        <div class="nav-list">
            <div class="nav-item active" data-section="home" onclick="showSection('home')"><span class="nav-icon">🏠</span> 首页</div>
            <div class="nav-item" data-section="data-center" onclick="showSection('data-center')"><span class="nav-icon">📚</span> 资料中心</div>
            <div class="nav-item" data-section="analysis" onclick="showSection('analysis')"><span class="nav-icon">🧠</span> AI分析中心</div>
            <div class="nav-item" data-section="interview" onclick="showSection('interview')"><span class="nav-icon">🎤</span> AI面试中心</div>
            <div class="nav-item" data-section="settings" onclick="showSection('settings')"><span class="nav-icon">⚙️</span> 系统设置</div>
        </div>
    </nav>

    <!-- 内容区 -->
    <main class="main-content">

    <!-- ===== 首页 ===== -->
    <section id="section-home" class="section-page active">
        <div class="welcome-banner">
            <h1>AI智能招聘平台</h1>
            <p>一站式简历分析、岗位匹配与智能面试平台 —— 基于 RAG 架构驱动</p>
        </div>
        <div class="stats-grid" id="homeStats"></div>
        <div class="card">
            <h3>📋 最近上传文件</h3>
            <div id="recentFiles"></div>
        </div>
    </section>

    <!-- ===== 资料中心 ===== -->
    <section id="section-data-center" class="section-page">
        <div class="card" style="margin-bottom:12px">
            <h3>📚 资料中心 <button class="btn btn-sm btn-success" onclick="openUploadModal()" style="margin-left:12px">📥 上传资料</button></h3>
            <p style="font-size:13px;color:#64748b">统一管理知识库、JD和简历文件，支持PDF/图片/文本</p>
        </div>
        <div class="card">
            <!-- 过滤栏 -->
            <div class="doc-filter-bar">
                <button class="doc-filter-btn active" onclick="filterDocs('all', this)">📋 全部<span class="filter-count" id="dcCountAll">0</span></button>
                <button class="doc-filter-btn" onclick="filterDocs('knowledge', this)">📖 知识库<span class="filter-count" id="dcCountKB">0</span></button>
                <button class="doc-filter-btn" onclick="filterDocs('jd', this)">💼 岗位JD<span class="filter-count" id="dcCountJD">0</span></button>
                <button class="doc-filter-btn" onclick="filterDocs('resume', this)">📄 简历<span class="filter-count" id="dcCountResume">0</span></button>
                <input class="input" id="dcSearch" placeholder="🔍 搜索文件名..." oninput="filterDocs(currentDCFilter)" style="margin-left:auto;width:220px">
            </div>
            <!-- 卡片网格 -->
            <div class="doc-grid" id="docCardGrid"></div>
            <div id="docEmpty" class="empty-state" style="display:none">
                <div class="empty-icon">📭</div>
                <div>暂无资料</div>
                <button class="empty-upload-btn" onclick="openUploadModal()">📥 上传第一份资料</button>
            </div>
        </div>
    </section>

    <!-- 上传资料模态窗 -->
    <div class="upload-modal-overlay" id="uploadModal" onclick="if(event.target===this)closeUploadModal()">
        <div class="upload-modal">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                <h3 style="margin:0">📥 上传资料</h3>
                <button class="modal-close" onclick="closeUploadModal()" style="font-size:22px">✕</button>
            </div>
            <!-- 资料类型选择 -->
            <div class="type-selector">
                <div class="type-option selected" id="uploadTypeKnowledge" onclick="selectUploadType('knowledge')">
                    <div class="type-icon">📖</div>
                    <div class="type-label">知识库资料</div>
                    <div class="type-desc">PDF/图片通用文档</div>
                </div>
                <div class="type-option" id="uploadTypeJD" onclick="selectUploadType('jd')">
                    <div class="type-icon">💼</div>
                    <div class="type-label">岗位JD</div>
                    <div class="type-desc">职位描述文档</div>
                </div>
                <div class="type-option" id="uploadTypeResume" onclick="selectUploadType('resume')">
                    <div class="type-icon">📄</div>
                    <div class="type-label">简历</div>
                    <div class="type-desc">候选人简历</div>
                </div>
            </div>
            <!-- 文件拖拽区 -->
            <div class="file-drop" id="uploadDropZone" onclick="document.getElementById('uploadFileInput').click()">
                <div class="drop-icon">📁</div>
                <div class="drop-text">点击选择文件或拖拽到此处</div>
                <div class="drop-hint">支持 PDF / PNG / JPG / JPEG / WEBP (后续扩展 DOCX / TXT / MD)</div>
                <div class="file-name" id="uploadFileName" style="display:none"></div>
            </div>
            <input type="file" id="uploadFileInput" accept=".pdf,.png,.jpg,.jpeg,.webp" style="display:none" onchange="handleFileSelect(event)">
            <div style="font-size:11px;color:#94a3b8;text-align:center" id="uploadStatus"></div>
            <div class="upload-actions">
                <button class="btn btn-outline" onclick="closeUploadModal()">取消</button>
                <button class="btn btn-primary" id="uploadSubmitBtn" onclick="doUnifiedUpload()" disabled>📤 上传</button>
            </div>
        </div>
    </div>

    <!-- 资料详情模态窗 -->
    <div class="modal-overlay" id="docDetailModal" onclick="if(event.target===this)closeDocDetail()">
        <div class="modal-box" id="docDetailBox"></div>
    </div>

    <!-- ===== AI分析中心 ===== -->
    <section id="section-analysis" class="section-page">
        <div class="card" style="margin-bottom:12px">
            <h3>🧠 AI分析中心</h3>
            <p style="font-size:13px;color:#64748b">AI问答、简历深度分析与岗位智能匹配</p>
        </div>
        <div class="card" style="padding:0 0 0 0">
            <!-- Tab 导航 -->
            <div class="tab-bar" id="anTabs" style="padding:0 24px">
                <button class="tab-btn active" onclick="switchAnTab('qa')">💬 AI问答</button>
                <button class="tab-btn" onclick="switchAnTab('resume')">📄 简历分析</button>
                <button class="tab-btn" onclick="switchAnTab('match')">🎯 岗位匹配分析</button>
            </div>
            <!-- Tab: AI问答 -->
            <div class="tab-panel active" id="an-qa" style="padding:0 24px 24px">
                <div class="action-row">
                    <button class="btn btn-sm btn-outline" onclick="qaSelectAll()">全选</button>
                    <button class="btn btn-sm btn-outline" onclick="qaDeselectAll()">取消全选</button>
                </div>
                <div class="pdf-checklist" id="qaChecklist"></div>
                <div class="action-row" style="margin-top:12px">
                    <input class="input" id="qaQuestion" placeholder="输入你的问题..." style="flex:1">
                    <button class="btn btn-primary" onclick="sendQA()">发送</button>
                    <button class="btn btn-success" onclick="doSummary()">总结选中PDF</button>
                </div>
                <div id="qaAnswer" style="display:none;margin-top:16px">
                    <div id="qaAnswerCard"></div>
                    <div id="qaSources"></div>
                </div>
            </div>
            <!-- Tab: 简历分析 -->
            <div class="tab-panel" id="an-resume" style="padding:0 24px 24px">
                <div class="action-row">
                    <select class="input" id="resumeSelect" style="min-width:280px"><option value="">-- 选择简历PDF --</option></select>
                    <button class="btn btn-primary" onclick="doResumeAnalysis()">开始分析</button>
                </div>
                <div id="resumeResult" style="display:none;margin-top:16px">
                    <div class="card"><div class="md-output" id="resumeOutput"></div></div>
                </div>
            </div>
            <!-- Tab: 岗位匹配分析 -->
            <div class="tab-panel" id="an-match" style="padding:0 24px 24px">
                <div class="match-layout">
                    <div class="card" style="margin-bottom:0">
                        <h4>🎯 结构化匹配</h4>
                        <div class="form-group">
                            <label>📄 选择已解析简历</label>
                            <select class="input w-full" id="matchResumeSel"><option value="">-- 请选择 --</option></select>
                        </div>
                        <div class="form-group">
                            <label>💼 选择JD</label>
                            <select class="input w-full" id="matchJdSel"><option value="">-- 请选择 --</option></select>
                        </div>
                        <button class="btn btn-primary btn-lg" onclick="doMatchV2()" style="width:100%">🚀 开始结构化匹配</button>
                        <div style="font-size:11px;color:#94a3b8;margin-top:8px;text-align:center">
                            技能(50%) + 项目(25%) + 学历(10%) + 经验(15%)
                        </div>
                    </div>
                    <div id="matchResult" style="display:none"></div>
                    <div id="matchEmpty" style="display:flex;align-items:center;justify-content:center;min-height:200px;color:#94a3b8;font-size:14px">
                        <div style="text-align:center">
                            <div style="font-size:48px;margin-bottom:12px">🔬</div>
                            <div>选择简历和JD，点击匹配按钮</div>
                            <div style="font-size:12px;color:#cbd5e1;margin-top:4px">简历需先在「资料中心→简历管理」中完成AI解析</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ===== AI面试中心 V3 ===== -->
    <section id="section-interview" class="section-page">
        <!-- ===== 创建面试页 ===== -->
        <div id="interview-create">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
                <!-- 步骤1+2：简历+JD -->
                <div class="card">
                    <h4>👤 步骤1：候选人</h4>
                    <div class="form-group">
                        <label>选择简历</label>
                        <select class="input w-full" id="ivResumeSel"><option value="">-- 请选择 --</option></select>
                    </div>
                    <div id="ivResumePreview" style="font-size:11px;color:#94a3b8;margin-top:8px;min-height:20px"></div>
                </div>
                <div class="card">
                    <h4>💼 步骤2：岗位</h4>
                    <div class="form-group">
                        <label>选择JD</label>
                        <select class="input w-full" id="ivJdSel"><option value="">-- 请选择 --</option></select>
                    </div>
                    <div id="ivJdPreview" style="font-size:11px;color:#94a3b8;margin-top:8px;min-height:20px"></div>
                </div>
            </div>
            <!-- 步骤3：面试模式 -->
            <div class="card">
                <h4>🎯 步骤3：选择面试模式</h4>
                <div class="mode-grid" id="ivModeGrid">
                    <div class="mode-card" data-mode="intern" onclick="selectMode('intern')">
                        <div class="mode-icon">🎓</div>
                        <div class="mode-name">实习冲刺模式</div>
                        <div class="mode-desc">适合实习生和初学者，验证项目真实性和基础能力</div>
                        <div class="mode-tags"><span>项目真实性验证</span><span>基础能力考察</span></div>
                    </div>
                    <div class="mode-card" data-mode="standard" onclick="selectMode('standard')">
                        <div class="mode-icon">📚</div>
                        <div class="mode-name">校招标准模式</div>
                        <div class="mode-desc">适合应届生，项目深挖+技术理解</div>
                        <div class="mode-tags"><span>项目深挖</span><span>技术理解</span></div>
                    </div>
                    <div class="mode-card" data-mode="bigtech" onclick="selectMode('bigtech')">
                        <div class="mode-icon">🏢</div>
                        <div class="mode-name">大厂挑战模式</div>
                        <div class="mode-desc">模拟字节/腾讯/阿里面试，连续追问+技术深挖</div>
                        <div class="mode-tags"><span>连续追问</span><span>技术深挖</span><span>高压</span></div>
                    </div>
                    <div class="mode-card" data-mode="pressure" onclick="selectMode('pressure')">
                        <div class="mode-icon">🔥</div>
                        <div class="mode-name">压力面模式</div>
                        <div class="mode-desc">高压提问+连环追问+质疑式面试</div>
                        <div class="mode-tags"><span>高压提问</span><span>连环追问</span><span>质疑式</span></div>
                    </div>
                </div>
            </div>
            <button class="btn btn-primary btn-lg" onclick="startInterview()" style="width:100%;margin-top:12px;padding:14px" id="ivStartBtn">🚀 开始面试</button>
        </div>

        <!-- ===== 面试进行中：V4 ===== -->
        <div id="interview-active" style="display:none">
            <!-- 顶部状态栏 -->
            <div class="iv-topbar" id="ivTopbar">
                <div class="iv-topbar-left">
                    <span class="iv-topbar-title" id="ivTopTitle">🎯 --</span>
                    <span class="iv-topbar-sep">|</span>
                    <span class="iv-topbar-sub" id="ivTopSub">📄 --</span>
                    <span class="iv-topbar-sep">|</span>
                    <span class="iv-topbar-mode" id="ivTopMode">--</span>
                </div>
                <div class="iv-topbar-progress">
                    <div class="iv-progress-label">
                        <span>第 <b id="ivProgCur">1</b> / <b id="ivProgMax">6</b> 题</span>
                        <span id="ivProgPct">0%</span>
                    </div>
                    <div class="iv-progress-track">
                        <div class="iv-progress-fill" id="ivProgFill" style="width:0%"></div>
                    </div>
                    <div class="iv-progress-eta" id="ivProgEta">预计剩余 12 分钟</div>
                </div>
                <div class="iv-topbar-actions">
                    <button class="btn btn-sm btn-outline" onclick="showHistory()">📋 记录</button>
                    <button class="btn btn-sm btn-danger" onclick="endInterview()">结束</button>
                </div>
            </div>

            <!-- 面试对话区 + 右侧仪表盘 -->
            <div class="iv-main-layout">
                <!-- 聊天区 -->
                <div class="iv-chat-area">
                    <div class="card iv-chat-card">
                        <div class="iv-chat-body" id="ivChatBody"></div>
                        <div class="iv-chat-footer">
                            <textarea id="ivAnswerInput" placeholder="输入你的回答...（Enter 发送，Shift+Enter 换行）" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();submitAnswer()}" rows="1"></textarea>
                            <button class="btn btn-primary" onclick="submitAnswer()" id="ivSendBtn">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 右侧仪表盘 -->
                <div class="iv-dashboard" id="ivDashboard">
                    <div class="card">
                        <div class="iv-dash-header">面试仪表盘</div>

                        <!-- 当前表现 -->
                        <div class="iv-dash-section">
                            <div class="iv-dash-section-title">当前表现</div>
                            <div class="iv-dash-perf" id="ivDashPerf">
                                <span class="iv-perf-dot" style="background:#94a3b8"></span>
                                <span>--</span>
                            </div>
                        </div>

                        <!-- 能力星级 -->
                        <div class="iv-dash-section">
                            <div class="iv-dash-section-title">能力评估</div>
                            <div class="iv-dash-star-row"><span>表达能力</span><span class="iv-stars" id="ivStarsExpr">☆☆☆☆☆</span></div>
                            <div class="iv-dash-star-row"><span>项目深度</span><span class="iv-stars" id="ivStarsProj">☆☆☆☆☆</span></div>
                            <div class="iv-dash-star-row"><span>岗位匹配</span><span class="iv-stars" id="ivStarsMatch">☆☆☆☆☆</span></div>
                        </div>

                        <!-- 风险提醒 -->
                        <div class="iv-dash-section" id="ivDashRisksSec" style="display:none">
                            <div class="iv-dash-section-title" style="color:#ef4444">⚠ 风险提醒</div>
                            <ul class="iv-dash-list" id="ivDashRisks"></ul>
                        </div>

                        <!-- 优势 -->
                        <div class="iv-dash-section" id="ivDashStrengthsSec" style="display:none">
                            <div class="iv-dash-section-title" style="color:#10b981">✅ 优势</div>
                            <ul class="iv-dash-list" id="ivDashStrengths"></ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 面试报告页 -->
        <div id="interview-report" style="display:none">
            <div style="max-width:780px;margin:0 auto">
                <div class="card" style="text-align:center;padding:32px 24px">
                    <div class="big-num" id="ivRptScore">--</div>
                    <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">综合得分</div>
                    <div style="font-size:24px;font-weight:700;color:#334155" id="ivRptMatch">--</div>
                    <div style="font-size:12px;color:#94a3b8">岗位匹配度</div>
                    <div style="margin-top:16px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap" id="ivRptStars"></div>
                </div>
                <div class="card" style="margin-top:12px">
                    <div class="md-output" id="ivReportContent"></div>
                </div>
                <button class="btn btn-primary btn-lg" onclick="resetInterview()" style="width:100%;margin-top:12px">🔄 开始新的面试</button>
            </div>
        </div>
    </section>

    <!-- ===== 系统设置 ===== -->
    <section id="section-settings" class="section-page">
        <div class="card">
            <h3>⚙️ 系统设置</h3>
            <div class="info-row"><span class="info-label">Embedding模型</span><span class="info-value">all-MiniLM-L6-v2 (SentenceTransformer)</span></div>
            <div class="info-row"><span class="info-label">向量数据库</span><span class="info-value">ChromaDB (Persistent)</span></div>
            <div class="info-row"><span class="info-label">LLM</span><span class="info-value">DeepSeek (deepseek-chat)</span></div>
            <div class="info-row"><span class="info-label">向量维度</span><span class="info-value">384</span></div>
            <div class="info-row"><span class="info-label">PDF上传目录</span><span class="info-value">./uploads/</span></div>
            <div class="info-row"><span class="info-label">向量库路径</span><span class="info-value">./chroma_db/</span></div>
        </div>
        <div class="card">
            <h3>📊 向量库状态</h3>
            <div class="stats-grid" id="settingsStats"></div>
        </div>
    </section>

    <!-- JD 详情弹窗 -->
    <div class="modal-overlay" id="jdModal" onclick="if(event.target===this)closeJDDetail()">
        <div class="modal-box" id="jdModalBox"></div>
    </div>

    </main>

    <script>
    "use strict";

    // ===== 简易 Markdown 渲染 =====
    function renderMd(text) {
        if (!text) return "";
        var h = text;
        h = h.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
        h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
        h = h.replace(/^# (.+)$/gm, "<h2>$1</h2>");
        h = h.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
        h = h.replace(/✅/g, "<span style='color:#166534'>✅</span>");
        h = h.replace(/❌/g, "<span style='color:#dc2626'>❌</span>");
        h = h.replace(/^---+/gm, "<hr>");
        h = h.replace(/^- (.+)$/gm, "<li>$1</li>");
        h = h.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");
        h = h.replace(/\n\n/g, "</p><p>");
        h = "<p>" + h + "</p>";
        h = h.replace(/<p><h/g, "<h").replace(/<\/h(\d)><\/p>/g, "</h$1>");
        h = h.replace(/<p><ul>/g, "<ul>").replace(/<\/ul><\/p>/g, "</ul>");
        h = h.replace(/<p><hr><\/p>/g, "<hr>");
        h = h.replace(/<p>\s*<\/p>/g, "");
        return h;
    }

    // ===== 全局数据 =====
    var pdfData = {};
    var docData = {};      // 统一资料中心数据
    var currentDCFilter = "all";   // 当前过滤类型

    // ===== 导航切换（单页应用） =====
    window.showSection = function(name) {
        // 隐藏所有 section
        var sections = document.querySelectorAll(".section-page");
        for (var i = 0; i < sections.length; i++) {
            sections[i].classList.remove("active");
        }
        // 显示目标 section
        var target = document.getElementById("section-" + name);
        if (target) {
            target.classList.add("active");
        }
        // 更新侧边栏高亮
        var navs = document.querySelectorAll(".nav-item");
        for (var j = 0; j < navs.length; j++) {
            navs[j].classList.remove("active");
        }
        var navItem = document.querySelector('.nav-item[data-section="' + name + '"]');
        if (navItem) {
            navItem.classList.add("active");
        }
        // 按需加载数据
        if (name === "home" || name === "settings") loadStats();
        if (name === "data-center") { refreshPDFData(); loadDocs(); }
        if (name === "analysis") { refreshAnalysisData(); loadDocs(); }
        if (name === "interview") { loadJDs(); loadResumes().then(function(){ refreshIVSelects(); }); loadDocs(); }
    };

    // ===== 资料中心 Tab 切换（保留兼容） =====
    window.switchDCTab = function(tab) {
        // 映射到新过滤系统
        if (tab === "kb") filterDocs("knowledge");
        else if (tab === "resume") filterDocs("resume");
        else if (tab === "jd") filterDocs("jd");
    };

    // ===== 统一资料中心 V2 =====
    window.loadDocs = async function() {
        try {
            var res = await fetch("/documents");
            docData = await res.json();
            filterDocs(currentDCFilter);
        } catch(e) { console.error("加载资料失败:", e); }
    };

    window.filterDocs = function(type, btn) {
        currentDCFilter = type;
        // 更新过滤按钮
        var btns = document.querySelectorAll(".doc-filter-btn");
        for (var i = 0; i < btns.length; i++) btns[i].classList.remove("active");
        if (btn) btn.classList.add("active");
        else {
            var autoBtn = document.querySelector('.doc-filter-btn[onclick*="' + type + '"]');
            if (autoBtn) autoBtn.classList.add("active");
        }
        renderDocCards();
    };

    window.renderDocCards = function() {
        var grid = document.getElementById("docCardGrid");
        var empty = document.getElementById("docEmpty");
        if (!grid || !empty) return;
        
        var keyword = (document.getElementById("dcSearch") ? document.getElementById("dcSearch").value : "").toLowerCase();
        var ids = Object.keys(docData);
        
        // 统计计数
        var countAll = 0, countKB = 0, countJD = 0, countResume = 0;
        for (var i = 0; i < ids.length; i++) {
            var t = docData[ids[i]].type || "knowledge";
            if (t === "knowledge") countKB++;
            else if (t === "jd") countJD++;
            else if (t === "resume") countResume++;
            countAll++;
        }
        var elAll = document.getElementById("dcCountAll"); if (elAll) elAll.textContent = countAll;
        var elKB = document.getElementById("dcCountKB"); if (elKB) elKB.textContent = countKB;
        var elJD = document.getElementById("dcCountJD"); if (elJD) elJD.textContent = countJD;
        var elResume = document.getElementById("dcCountResume"); if (elResume) elResume.textContent = countResume;
        
        var filtered = [];
        for (var j = 0; j < ids.length; j++) {
            var id = ids[j];
            var d = docData[id];
            if (currentDCFilter !== "all" && d.type !== currentDCFilter) continue;
            if (keyword && d.filename.toLowerCase().indexOf(keyword) === -1) continue;
            filtered.push({id: id, data: d});
        }
        
        if (filtered.length === 0) {
            grid.innerHTML = "";
            empty.style.display = "block";
            if (ids.length === 0) {
                empty.innerHTML = '<div class="empty-icon">📭</div><div>暂无资料</div><button class="empty-upload-btn" onclick="openUploadModal()">📥 上传第一份资料</button>';
            }
            return;
        }
        empty.style.display = "none";
        
        // 类型图标和颜色
        var typeIcons = {"knowledge": "📖", "jd": "💼", "resume": "📄"};
        var typeLabels = {"knowledge": "知识库", "jd": "岗位JD", "resume": "简历"};
        var barClass = {"knowledge": "bar-knowledge", "jd": "bar-jd", "resume": "bar-resume"};
        var typeTagClass = {"knowledge": "type-knowledge", "jd": "type-jd", "resume": "type-resume"};
        var srcClass = {"pdf": "src-pdf", "image": "src-image", "text": "src-text"};
        var srcIcon = {"pdf": "📕", "image": "🖼", "text": "📝"};
        
        var html = "";
        for (var k = 0; k < filtered.length; k++) {
            var item = filtered[k];
            var d = item.data;
            var id = item.id;
            var dtype = d.type || "knowledge";
            var stype = d.source_type || "pdf";
            var barCls = barClass[dtype] || "bar-knowledge";
            var tagCls = typeTagClass[dtype] || "type-knowledge";
            var srcCls = srcClass[stype] || "src-pdf";
            
            html += '<div class="doc-card" onclick="openDocDetail(\'' + id + '\')">';
            html += '<div class="doc-bar ' + barCls + '"></div>';
            html += '<div class="doc-icon">' + (typeIcons[dtype] || "📄") + '</div>';
            html += '<div class="doc-filename" title="' + d.filename + '">' + d.filename + '</div>';
            html += '<div class="doc-tags">';
            html += '<span class="doc-tag ' + tagCls + '">' + (typeLabels[dtype] || dtype) + '</span>';
            html += '<span class="doc-source-tag ' + srcCls + '">' + (srcIcon[stype] || "") + " " + stype.toUpperCase() + '</span>';
            html += '</div>';
            html += '<div class="doc-meta">';
            html += '<span>📅 ' + (d.upload_time || "").substring(0, 16) + '</span>';
            html += '<span>🧩 ' + (d.chunk_count || 0) + ' chunks</span>';
            html += '<span>📏 ' + (d.text_length || 0) + ' 字</span>';
            html += '</div>';
            html += '<div class="doc-actions" onclick="event.stopPropagation()">';
            html += '<button class="btn btn-sm btn-outline" onclick="openDocDetail(\'' + id + '\')">🔍 详情</button>';
            html += '<button class="btn btn-sm btn-danger" onclick="deleteDocument(\'' + id + '\')">🗑 删除</button>';
            html += '</div>';
            html += '</div>';
        }
        grid.innerHTML = html;
    };

    // ===== 上传模态窗 =====
    var uploadSelectedType = "knowledge";
    var uploadSelectedFile = null;

    window.openUploadModal = function(type) {
        if (type) uploadSelectedType = type;
        uploadSelectedFile = null;
        document.getElementById("uploadFileInput").value = "";
        document.getElementById("uploadFileName").style.display = "none";
        document.getElementById("uploadSubmitBtn").disabled = true;
        document.getElementById("uploadStatus").textContent = "";
        document.getElementById("uploadDropZone").classList.remove("has-file");
        selectUploadTypeUI(uploadSelectedType);
        document.getElementById("uploadModal").classList.add("active");
    };

    window.closeUploadModal = function() {
        document.getElementById("uploadModal").classList.remove("active");
    };

    window.selectUploadType = function(type) {
        uploadSelectedType = type;
        selectUploadTypeUI(type);
    };

    function selectUploadTypeUI(type) {
        var ids = ["uploadTypeKnowledge", "uploadTypeJD", "uploadTypeResume"];
        var vals = ["knowledge", "jd", "resume"];
        for (var i = 0; i < ids.length; i++) {
            var el = document.getElementById(ids[i]);
            if (!el) continue;
            if (vals[i] === type) el.classList.add("selected");
            else el.classList.remove("selected");
        }
    }

    window.handleFileSelect = function(event) {
        var file = event.target.files[0];
        if (!file) return;
        uploadSelectedFile = file;
        var fn = document.getElementById("uploadFileName");
        fn.textContent = "📎 " + file.name + " (" + formatSize(file.size) + ")";
        fn.style.display = "block";
        document.getElementById("uploadDropZone").classList.add("has-file");
        document.getElementById("uploadSubmitBtn").disabled = false;
        document.getElementById("uploadStatus").textContent = "";
    };

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // 拖拽上传
    (function() {
        setTimeout(function() {
            var zone = document.getElementById("uploadDropZone");
            if (!zone) return;
            zone.addEventListener("dragover", function(e) { e.preventDefault(); zone.style.borderColor = "#0ea5e9"; zone.style.background = "#f8fafc"; });
            zone.addEventListener("dragleave", function(e) { e.preventDefault(); zone.style.borderColor = ""; zone.style.background = ""; });
            zone.addEventListener("drop", function(e) {
                e.preventDefault();
                zone.style.borderColor = ""; zone.style.background = "";
                var file = e.dataTransfer.files[0];
                if (!file) return;
                uploadSelectedFile = file;
                var fn = document.getElementById("uploadFileName");
                fn.textContent = "📎 " + file.name + " (" + formatSize(file.size) + ")";
                fn.style.display = "block";
                zone.classList.add("has-file");
                document.getElementById("uploadSubmitBtn").disabled = false;
            });
        }, 500);
    })();

    window.doUnifiedUpload = async function() {
        if (!uploadSelectedFile) { alert("请选择文件"); return; }
        var btn = document.getElementById("uploadSubmitBtn");
        var statusEl = document.getElementById("uploadStatus");
        btn.disabled = true; btn.textContent = "⏳ 上传中...";
        statusEl.textContent = "正在上传并处理...";
        
        var fd = new FormData();
        fd.append("file", uploadSelectedFile);
        
        try {
            var res = await fetch("/documents/upload?document_type=" + uploadSelectedType, { method: "POST", body: fd });
            var d = await res.json();
            if (d.success) {
                statusEl.textContent = "✅ 上传成功！" + d.document.chunk_count + " chunks";
                btn.textContent = "📤 上传";
                // 刷新
                await loadDocs();
                await refreshPDFData();
                loadStats();
                // 延迟关闭
                setTimeout(function() { closeUploadModal(); }, 800);
            } else {
                statusEl.textContent = "❌ " + (d.message || "上传失败");
                btn.disabled = false; btn.textContent = "📤 上传";
            }
        } catch(e) {
            statusEl.textContent = "❌ 网络错误: " + e;
            btn.disabled = false; btn.textContent = "📤 上传";
        }
    };

    // ===== 删除资料 =====
    window.deleteDocument = async function(docId) {
        var info = docData[docId];
        var fname = info ? info.filename : docId;
        if (!confirm("确定删除 " + fname + " 吗？\n此操作将同时删除向量数据和缓存，不可恢复。")) return;
        await fetch("/documents/" + docId, { method: "DELETE" });
        await loadDocs();
        await refreshPDFData();
        loadStats();
    };

    // ===== 资料详情模态窗 =====
    window.openDocDetail = async function(docId) {
        try {
            var res = await fetch("/documents/" + docId);
            var d = await res.json();
            if (d.error) { alert(d.error); return; }
            
            var typeLabels = {"knowledge": "知识库", "jd": "岗位JD", "resume": "简历"};
            var typeIcons = {"knowledge": "📖", "jd": "💼", "resume": "📄"};
            var srcIcons = {"pdf": "📕 PDF", "image": "🖼 图片", "text": "📝 文本"};
            
            var box = document.getElementById("docDetailBox");
            var html = '<div class="modal-header"><h3>' + (typeIcons[d.type] || "📄") + " " + d.filename + '</h3><button class="modal-close" onclick="closeDocDetail()">✕</button></div>';
            html += '<div class="modal-section">';
            html += '<div class="info-row"><span class="info-label">资料类型</span><span class="info-value">' + (typeLabels[d.type] || d.type) + '</span></div>';
            html += '<div class="info-row"><span class="info-label">来源类型</span><span class="info-value">' + (srcIcons[d.source_type] || d.source_type) + '</span></div>';
            html += '<div class="info-row"><span class="info-label">上传时间</span><span class="info-value">' + (d.upload_time || "") + '</span></div>';
            html += '<div class="info-row"><span class="info-label">Chunk数量</span><span class="info-value">' + (d.chunk_count || 0) + '</span></div>';
            html += '<div class="info-row"><span class="info-label">文本长度</span><span class="info-value">' + (d.text_length || 0) + ' 字符</span></div>';
            html += '<div class="info-row"><span class="info-label">文件大小</span><span class="info-value">' + formatSize(d.file_size || 0) + '</span></div>';
            if (d.ocr_engine) html += '<div class="info-row"><span class="info-label">OCR引擎</span><span class="info-value">' + d.ocr_engine + ' (' + d.ocr_time_ms + 'ms)</span></div>';
            html += '<div class="info-row"><span class="info-label">向量耗时</span><span class="info-value">' + (d.embed_time_ms || 0) + 'ms</span></div>';
            html += '</div>';
            
            // Chunk列表
            if (d.chunks && d.chunks.length > 0) {
                html += '<div class="modal-section"><h4>📑 Chunk列表 (' + d.chunks.length + ')</h4><div class="chunk-list">';
                for (var i = 0; i < d.chunks.length; i++) {
                    var c = d.chunks[i];
                    html += '<div class="chunk-item">';
                    html += '<div class="chunk-header"><span class="chunk-num">Chunk #' + c.index + '</span><span class="chunk-len">' + c.length + '字</span></div>';
                    html += '<div class="chunk-preview">' + c.preview + '</div>';
                    html += '<div class="chunk-full" id="chunkFull' + c.index + '">' + escapeHtml(c.full_text) + '</div>';
                    html += '<span class="chunk-toggle" onclick="toggleChunk(' + c.index + ')">📖 展开完整内容</span>';
                    html += '</div>';
                }
                html += '</div></div>';
            }
            
            html += '<div class="action-row" style="justify-content:flex-end">';
            html += '<button class="btn btn-danger" onclick="deleteDocument(\'' + d.id + '\');closeDocDetail()">🗑 删除此资料</button>';
            html += '</div>';
            
            box.innerHTML = html;
            document.getElementById("docDetailModal").classList.add("active");
        } catch(e) { alert("获取详情失败: " + e); }
    };

    window.closeDocDetail = function() {
        document.getElementById("docDetailModal").classList.remove("active");
    };

    window.toggleChunk = function(index) {
        var el = document.getElementById("chunkFull" + index);
        var toggleEl = el.previousElementSibling;
        if (el.style.display === "block") {
            el.style.display = "none";
            if (toggleEl) toggleEl.textContent = "📖 展开完整内容";
        } else {
            el.style.display = "block";
            if (toggleEl) toggleEl.textContent = "📕 收起";
        }
    };

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // ===== 兼容旧函数 =====
    window.refreshPDFData = async function() {
        try {
            var res = await fetch("/pdfs");
            pdfData = await res.json();
            refreshSelects();
            refreshQAChecklist();
            renderRecentFiles();
        } catch(e) { console.error("加载PDF数据失败:", e); }
    };

    // 保留旧renderDCTable兼容
    function renderDCTable(type) { /* 已迁移到统一renderDocCards */ }

    function refreshDCAll() {
        loadDocs();
    }

    window.uploadToDC = async function(type) {
        // 重定向到统一上传
        openUploadModal(type === "resume" ? "resume" : type === "jd" ? "jd" : "knowledge");
    };

    window.deletePdf = async function(id) {
        // 统一删除
        await deleteDocument(id);
    };

    // ===== AI分析中心 Tab 切换 =====
    window.switchAnTab = function(tab) {
        var btns = document.querySelectorAll("#anTabs .tab-btn");
        for (var i = 0; i < btns.length; i++) btns[i].classList.remove("active");
        var btn = document.querySelector('#anTabs .tab-btn[onclick*="' + tab + '"]');
        if (btn) btn.classList.add("active");
        var panels = ["qa","resume","match"];
        for (var j = 0; j < panels.length; j++) {
            var p = document.getElementById("an-" + panels[j]);
            if (p) p.classList.remove("active");
        }
        var target = document.getElementById("an-" + tab);
        if (target) target.classList.add("active");
        if (tab === "qa") refreshQAChecklist();
        if (tab === "resume") refreshSelects();
        if (tab === "match") { loadResumes(); refreshMatchV2Selects(); }
    };

    function refreshAnalysisData() {
        refreshQAChecklist();
        refreshSelects();
    }

    // ===== 首页 =====
    function renderRecentFiles() {
        // 从统一docData获取最近文件，回退到pdfData
        var ids = Object.keys(docData);
        if (ids.length === 0) ids = Object.keys(pdfData);
        var el = document.getElementById("recentFiles");
        if (!el) return;
        if (ids.length === 0) { el.innerHTML = '<div class="empty-state"><div>暂无文件</div></div>'; return; }
        var html = '<table><thead><tr><th>文件名</th><th style="width:100px">类型</th><th style="width:100px">Chunks</th></tr></thead><tbody>';
        var recent = ids.slice(-5).reverse();
        var typeLabels = {"knowledge": "📖 知识库", "jd": "💼 JD", "resume": "📄 简历"};
        for (var i = 0; i < recent.length; i++) {
            var id = recent[i];
            var info = docData[id] || pdfData[id] || {};
            var fname = info.filename || "";
            var dtype = info.type || "knowledge";
            var chunks = info.chunk_count || info.chunks || 0;
            html += '<tr><td>' + fname + '</td><td>' + (typeLabels[dtype] || dtype) + '</td><td><span class="badge badge-info">' + chunks + '</span></td></tr>';
        }
        html += '</tbody></table>';
        el.innerHTML = html;
    }

    window.loadStats = async function() {
        try {
            var res = await fetch("/stats");
            var d = await res.json();
            var qaCount = parseInt(localStorage.getItem("qa_count") || "0");
            var ivHistory = [];
            try { var r2 = await fetch("/interview/history"); ivHistory = await r2.json(); } catch(e) {}
            var ivCount = ivHistory.length;
            var homeCards = document.getElementById("homeStats");
            if (homeCards) {
                homeCards.innerHTML =
                    '<div class="stat-card"><div class="stat-num">' + d.pdf_count + '</div><div class="stat-label">资料总数</div></div>' +
                    '<div class="stat-card accent"><div class="stat-num">' + d.chunk_count + '</div><div class="stat-label">Chunk 数量</div></div>' +
                    '<div class="stat-card warn"><div class="stat-num">' + d.vector_count + '</div><div class="stat-label">向量索引</div></div>' +
                    '<div class="stat-card purple"><div class="stat-num">' + qaCount + '</div><div class="stat-label">AI问答次数</div></div>' +
                    '<div class="stat-card rose"><div class="stat-num">' + ivCount + '</div><div class="stat-label">面试次数</div></div>';
            }
            var ss = document.getElementById("settingsStats");
            if (ss) {
                ss.innerHTML =
                    '<div class="stat-card"><div class="stat-num">' + d.pdf_count + '</div><div class="stat-label">资料总数</div></div>' +
                    '<div class="stat-card accent"><div class="stat-num">' + (d.kb_count || 0) + '</div><div class="stat-label">知识库</div></div>' +
                    '<div class="stat-card warn"><div class="stat-num">' + (d.jd_count || 0) + '</div><div class="stat-label">岗位JD</div></div>' +
                    '<div class="stat-card purple"><div class="stat-num">' + (d.resume_count || 0) + '</div><div class="stat-label">简历</div></div>' +
                    '<div class="stat-card rose"><div class="stat-num">' + d.vector_count + '</div><div class="stat-label">向量总数</div></div>';
            }
        } catch(e) { console.error("加载统计失败:", e); }
    };

    // ===== AI问答 =====
    function refreshQAChecklist() {
        var container = document.getElementById("qaChecklist");
        if (!container) return;
        // 从统一docData获取知识库类型的资料
        var ids = Object.keys(docData);
        var kbItems = [];
        for (var i = 0; i < ids.length; i++) {
            if (docData[ids[i]].type === "knowledge") kbItems.push({id: ids[i], name: docData[ids[i]].filename});
        }
        // 兼容：如果docData为空，回退到pdfData
        if (kbItems.length === 0) {
            ids = Object.keys(pdfData);
            for (var j = 0; j < ids.length; j++) {
                kbItems.push({id: ids[j], name: pdfData[ids[j]].filename});
            }
        }
        if (kbItems.length === 0) { container.innerHTML = '<div class="empty-state" style="padding:20px"><div>暂无知识库资料，请先上传</div></div>'; return; }
        var html = "";
        for (var k = 0; k < kbItems.length; k++) {
            html += '<label><input type="checkbox" value="' + kbItems[k].id + '"> ' + kbItems[k].name + '</label>';
        }
        container.innerHTML = html;
    }

    window.qaSelectAll = function() {
        var cbs = document.querySelectorAll("#qaChecklist input[type=checkbox]");
        for (var i = 0; i < cbs.length; i++) cbs[i].checked = true;
    };

    window.qaDeselectAll = function() {
        var cbs = document.querySelectorAll("#qaChecklist input[type=checkbox]");
        for (var i = 0; i < cbs.length; i++) cbs[i].checked = false;
    };

    function getQAChecked() {
        var cbs = document.querySelectorAll("#qaChecklist input[type=checkbox]:checked");
        var ids = [];
        for (var i = 0; i < cbs.length; i++) ids.push(cbs[i].value);
        return ids;
    }

    window.sendQA = async function() {
        var checked = getQAChecked();
        if (checked.length === 0) { alert("请勾选要检索的PDF"); return; }
        var q = document.getElementById("qaQuestion").value.trim();
        if (!q) { alert("请输入问题"); return; }
        document.getElementById("qaAnswer").style.display = "block";
        document.getElementById("qaAnswerCard").innerHTML = '<div class="loading">⏳ 检索中...</div>';
        document.getElementById("qaSources").innerHTML = "";
        var res = await fetch("/chat", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ collection_names: checked, question: q, mode: "qa" }) });
        var d = await res.json();
        document.getElementById("qaAnswerCard").innerHTML = '<div class="answer-card"><div class="answer-label">AI 回答</div><div class="answer-text">' + d.answer + '</div></div>';
        renderSources(d.sources);
        // 记录问答次数
        var count = parseInt(localStorage.getItem("qa_count") || "0") + 1;
        localStorage.setItem("qa_count", count);
        loadStats();
    };

    window.doSummary = async function() {
        var checked = getQAChecked();
        if (checked.length === 0) { alert("请勾选PDF"); return; }
        document.getElementById("qaAnswer").style.display = "block";
        document.getElementById("qaAnswerCard").innerHTML = '<div class="loading">⏳ 总结中...</div>';
        document.getElementById("qaSources").innerHTML = "";
        var res = await fetch("/chat", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ collection_names: checked, question: "请总结全文", mode: "summary" }) });
        var d = await res.json();
        document.getElementById("qaAnswerCard").innerHTML = '<div style="white-space:pre-wrap;line-height:1.8;font-size:14px;color:#334155">' + d.answer + '</div>';
    };

    function renderSources(sources) {
        var el = document.getElementById("qaSources");
        if (!el) return;
        if (!sources || sources.length === 0) { el.innerHTML = ""; return; }
        var files = {};
        for (var i = 0; i < sources.length; i++) {
            var f = sources[i].filename;
            if (!files[f]) files[f] = [];
            files[f].push(sources[i]);
        }
        var fnames = Object.keys(files);
        var html = '<div class="sources-header">📎 引用来源<span class="count-badge">' + sources.length + '</span></div>';
        for (var fi = 0; fi < fnames.length; fi++) {
            var fname = fnames[fi], items = files[fname];
            for (var si = 0; si < items.length; si++) {
                var src = items[si];
                var simClass = "low";
                if (src.similarity != null) { if (src.similarity >= 70) simClass = "high"; else if (src.similarity >= 40) simClass = "mid"; }
                var preview = src.content.substring(0, 100);
                if (src.content.length > 100) preview += "...";
                var uid = "src_" + src.rank + "_" + Math.random().toString(36).substr(2, 6);
                var ctxHtml = "";
                if (src.prev_chunk) ctxHtml += '<div class="ctx-label">⬆ 上一段</div><div style="color:#94a3b8;margin-bottom:8px">' + src.prev_chunk + '</div>';
                ctxHtml += '<div class="ctx-label">📍 命中段落（来源 #' + src.rank + '）</div><div class="ctx-cur">' + src.content + '</div>';
                if (src.next_chunk) ctxHtml += '<div class="ctx-label" style="margin-top:8px">⬇ 下一段</div><div style="color:#94a3b8">' + src.next_chunk + '</div>';
                html +=
                    '<div class="source-card" id="' + uid + '">' +
                        '<div class="src-meta">' +
                            '<span class="src-file">📄 ' + fname + '</span>' +
                            '<span class="src-cid">' + src.collection + '</span>' +
                        '</div>' +
                        '<div style="display:flex;gap:12px;margin-bottom:8px;font-size:11px;color:#94a3b8">' +
                            '<span>📏 Dist: ' + (src.distance != null ? src.distance : '-') + '</span>' +
                            '<span>🔍 向量: ' + (src.vector_score != null ? src.vector_score : '0') + '%</span>' +
                            '<span>📝 BM25: ' + (src.bm25_score != null ? src.bm25_score : '0') + '%</span>' +
                            '<span>🔀 Hybrid: ' + (src.hybrid_score != null ? src.hybrid_score : '0') + '%</span>' +
                        '</div>' +
                        '<div style="font-size:13px;font-weight:700;color:#0ea5e9;margin-bottom:8px">' +
                            '🎯 Cross Score: ' + (src.cross_score != null ? src.cross_score : '0') + '%' +
                        '</div>' +
                        '<div class="src-preview" id="' + uid + '_preview">' + preview + '</div>' +
                        '<div class="src-expanded" id="' + uid + '_full">' + ctxHtml + '</div>' +
                        '<div class="src-toggle" onclick="toggleSource(\'' + uid + '\')" id="' + uid + '_btn">展开全文 ▾</div>' +
                    '</div>';
            }
        }
        el.innerHTML = html;
    }

    window.toggleSource = function(uid) {
        var expanded = document.getElementById(uid + "_full");
        var preview = document.getElementById(uid + "_preview");
        var btn = document.getElementById(uid + "_btn");
        if (expanded.style.display === "block") {
            expanded.style.display = "none";
            preview.style.display = "block";
            btn.textContent = "展开全文 ▾";
        } else {
            expanded.style.display = "block";
            preview.style.display = "none";
            btn.textContent = "收起 △";
        }
    };

    // ===== 简历分析 =====
    window.doResumeAnalysis = async function() {
        var id = document.getElementById("resumeSelect").value;
        if (!id) { alert("请选择简历PDF"); return; }
        document.getElementById("resumeResult").style.display = "block";
        document.getElementById("resumeOutput").innerHTML = '<div class="loading">⏳ 深度分析中，预计20~40秒...</div>';
        var res = await fetch("/chat", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
            collection_names: [id],
            question: "你是一名资深技术面试官。请从以下维度分析该简历：\n\n# 技术栈分析\n分析掌握程度\n\n# 项目经验分析\n分析项目质量\n\n# 优势\n列出3~5点\n\n# 不足\n列出3~5点\n\n# 岗位匹配度\n适合哪些岗位\n\n# 简历优化建议\n给出具体修改建议\n\n# 综合评分\n100分制\n\n# 面试通过概率\n给出百分比\n\n请使用Markdown格式输出。",
            mode: "summary"
        }) });
        var d = await res.json();
        document.getElementById("resumeOutput").innerHTML = renderMd(d.answer);
    };

    // ===== 岗位匹配 =====
    window.doJobMatch = async function() {
        var rid = document.getElementById("matchResumeSel").value;
        var jid = document.getElementById("matchJdSel").value;
        if (!rid) { alert("请选择简历"); return; }
        if (!jid) { alert("请选择JD"); return; }
        if (rid === jid) { alert("简历和JD不能选择同一个PDF"); return; }
        document.getElementById("matchResult").style.display = "block";
        document.getElementById("matchOutput").innerHTML = '<div class="loading">⏳ 深度分析中，预计20~40秒...</div>';
        var res = await fetch("/job_match", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ resume_collection: rid, jd_collection: jid }) });
        var d = await res.json();
        document.getElementById("matchOutput").innerHTML = renderMd(d.answer);
    };

    // ===== 下拉框刷新 =====
    function refreshSelects() {
        var ids = Object.keys(pdfData);
        var selects = [document.getElementById("resumeSelect"), document.getElementById("matchResumeSel"), document.getElementById("matchJdSel")];
        for (var si = 0; si < selects.length; si++) {
            var sel = selects[si];
            if (!sel) continue;
            var cur = sel.value;
            sel.innerHTML = '<option value="">-- 请选择 --</option>';
            for (var i = 0; i < ids.length; i++) {
                var opt = document.createElement("option");
                opt.value = ids[i];
                opt.textContent = pdfData[ids[i]].filename;
                sel.appendChild(opt);
            }
            if (cur) {
                var found = false;
                for (var j = 0; j < ids.length; j++) { if (ids[j] === cur) { found = true; break; } }
                if (found) sel.value = cur;
            }
        }
    }

    // ===== 初始化 =====
    window.onload = function() {
        refreshPDFData();
        loadDocs();
        loadJDs();
        loadStats();
        showSection("home");
    };

    // ===== JD管理 =====
    var jdData = {};

    async function loadJDs() {
        try {
            var res = await fetch("/jd/list");
            jdData = await res.json();
            renderJDCards();
        } catch(e) { console.error("加载JD失败:", e); }
    }

    function renderJDCards() {
        var grid = document.getElementById("jdCardGrid");
        var empty = document.getElementById("jdEmpty");
        var countEl = document.getElementById("jdCount");
        if (!grid || !empty) return;
        var ids = Object.keys(jdData);
        if (countEl) countEl.textContent = "共 " + ids.length + " 条JD";
        if (ids.length === 0) {
            grid.innerHTML = "";
            empty.style.display = "block";
            return;
        }
        empty.style.display = "none";
        var html = "";
        for (var i = 0; i < ids.length; i++) {
            var jd = jdData[ids[i]];
            var skillsHtml = "";
            if (jd.core_skills && jd.core_skills.length > 0) {
                for (var s = 0; s < Math.min(jd.core_skills.length, 4); s++) {
                    skillsHtml += '<span class="jd-tag">' + jd.core_skills[s] + '</span>';
                }
                if (jd.core_skills.length > 4) skillsHtml += '<span class="jd-tag">+' + (jd.core_skills.length - 4) + '</span>';
            }
            var title = jd.title || "未命名JD";
            var company = jd.company || "未填写公司";
            var edu = jd.education || "-";
            var exp = jd.experience || "-";
            var quality = jd.quality_score || 0;
            var grade, gradeClass;
            if (quality >= 90) { grade = 'S'; gradeClass = 'grade-s'; }
            else if (quality >= 70) { grade = 'A'; gradeClass = 'grade-a'; }
            else if (quality >= 50) { grade = 'B'; gradeClass = 'grade-b'; }
            else { grade = 'C'; gradeClass = 'grade-c'; }
            var sourceClass = jd.source === "pdf" ? "pdf" : "text";
            var sourceLabel = jd.source === "pdf" ? "PDF" : "文本";
            html +=
                '<div class="jd-card" onclick="openJDDetail(\'' + jd.id + '\')">' +
                    '<span class="jd-source ' + sourceClass + '">' + sourceLabel + '</span>' +
                    '<span class="jd-grade-badge ' + gradeClass + '">' + grade + '级</span>' +
                    '<div class="jd-title">' + title + '</div>' +
                    '<div class="jd-company">🏢 ' + company + '</div>' +
                    '<div class="jd-meta">' +
                        '<span>🎓 ' + edu + '</span>' +
                        '<span>💼 ' + exp + '</span>' +
                    '</div>' +
                    '<div class="jd-tags">' + (skillsHtml || '<span style="font-size:11px;color:#94a3b8">暂无技能</span>') + '</div>' +
                    '<div class="jd-actions" onclick="event.stopPropagation()">' +
                        '<button class="btn btn-sm btn-primary" onclick="openJDDetail(\'' + jd.id + '\')">详情</button>' +
                        '<button class="btn btn-sm btn-outline" onclick="openJDEdit(\'' + jd.id + '\')">编辑</button>' +
                        '<button class="btn btn-sm btn-danger" onclick="doDeleteJD(\'' + jd.id + '\')">删除</button>' +
                    '</div>' +
                '</div>';
        }
        grid.innerHTML = html;
    }

    window.showJDForm = function(type) {
        if (type === "create") {
            document.getElementById("jdFormCreate").style.display = "block";
            document.getElementById("jdFormUpload").style.display = "none";
            document.getElementById("jdFormTitle").value = "";
            document.getElementById("jdFormCompany").value = "";
            document.getElementById("jdFormContent").value = "";
        } else {
            document.getElementById("jdFormUpload").style.display = "block";
            document.getElementById("jdFormCreate").style.display = "none";
            document.getElementById("jdPdfFile").value = "";
        }
    };

    window.hideJDForm = function(type) {
        if (type === "create") document.getElementById("jdFormCreate").style.display = "none";
        else document.getElementById("jdFormUpload").style.display = "none";
    };

    window.doCreateJD = async function() {
        var title = document.getElementById("jdFormTitle").value.trim();
        var company = document.getElementById("jdFormCompany").value.trim();
        var content = document.getElementById("jdFormContent").value.trim();
        if (!content) { alert("请填写JD内容"); return; }
        var btn = event.target;
        btn.disabled = true;
        btn.textContent = "⏳ 保存并解析中...";
        try {
            var res = await fetch("/jd/create", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({title: title, company: company, content: content})
            });
            var d = await res.json();
            alert("JD创建成功！AI已自动解析关键信息。");
            hideJDForm("create");
            await loadJDs();
        } catch(e) { alert("创建失败: " + e); }
        btn.disabled = false;
        btn.textContent = "💾 保存并AI解析";
    };

    window.doUploadJD = async function() {
        var fileInput = document.getElementById("jdPdfFile");
        var file = fileInput.files[0];
        if (!file) { alert("请选择文件（PDF或图片）"); return; }
        var fd = new FormData(); fd.append("file", file);
        var btn = event.target;
        btn.disabled = true;
        btn.textContent = "⏳ 上传解析中...";
        try {
            var res = await fetch("/jd/upload_pdf", { method: "POST", body: fd });
            var d = await res.json();
            if (d.jd) {
                alert("JD上传并解析成功！");
                hideJDForm("upload");
                await loadJDs();
            } else {
                alert(d.message || "上传失败");
            }
        } catch(e) { alert("上传失败: " + e); }
        btn.disabled = false;
        btn.textContent = "上传并解析";
    };

    window.openJDDetail = function(id) {
        var jd = jdData[id];
        if (!jd) return;

        // 技能标签（最多5个+N展开）
        var renderSkills = function(skills, cls) {
            if (!skills || skills.length === 0) return '<span style="font-size:12px;color:#94a3b8">暂无</span>';
            var show = Math.min(skills.length, 5);
            var html = '';
            for (var i = 0; i < show; i++) html += '<span class="' + cls + '">' + skills[i] + '</span>';
            if (skills.length > 5) {
                html += '<span class="' + cls + ' skill-more" onclick="event.stopPropagation();toggleSkills(this)" data-skills="' + encodeURIComponent(JSON.stringify(skills)) + '" data-cls="' + cls + '">+' + (skills.length - 5) + '</span>';
            }
            return html;
        };

        var coreHtml = renderSkills(jd.core_skills, 'core-tag');
        var bonusHtml = renderSkills(jd.bonus_skills, 'bonus-tag');

        var benefitsHtml = "";
        if (jd.benefits && jd.benefits.length > 0) {
            benefitsHtml = '<div class="modal-section"><h4>🎁 企业福利</h4><div class="skill-tags">';
            for (var b = 0; b < jd.benefits.length; b++) {
                benefitsHtml += '<span style="padding:4px 10px;border-radius:8px;font-size:12px;font-weight:500;background:#f0fdf4;color:#166534">' + jd.benefits[b] + '</span>';
            }
            benefitsHtml += '</div></div>';
        }

        var focusHtml = "";
        if (jd.interview_focus && jd.interview_focus.length > 0) {
            focusHtml = '<div class="modal-section"><h4>🎯 面试重点</h4><ul class="focus-list">';
            for (var f = 0; f < jd.interview_focus.length; f++) {
                focusHtml += '<li>' + jd.interview_focus[f] + '</li>';
            }
            focusHtml += '</ul></div>';
        }

        var quality = jd.quality_score || 0;
        var grade, gradeClass;
        if (quality >= 90) { grade = 'S'; gradeClass = 'grade-s'; }
        else if (quality >= 70) { grade = 'A'; gradeClass = 'grade-a'; }
        else if (quality >= 50) { grade = 'B'; gradeClass = 'grade-b'; }
        else { grade = 'C'; gradeClass = 'grade-c'; }

        var gradInfo = jd.graduation_year ? ' · 🎓 ' + jd.graduation_year + '届' : '';

        var box = document.getElementById("jdModalBox");
        box.innerHTML =
            '<div class="modal-header">' +
                '<div style="flex:1">' +
                    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">' +
                        '<h3 style="margin:0">' + (jd.title || "未命名JD") + '</h3>' +
                        '<span class="quality-badge ' + gradeClass + '">' + grade + '级 ' + quality + '分</span>' +
                    '</div>' +
                    '<div style="font-size:13px;color:#64748b">🏢 ' + (jd.company || "未填写") + ' · ' + (jd.source === "pdf" ? "PDF" : "文本") + gradInfo + '</div>' +
                '</div>' +
                '<button class="modal-close" onclick="closeJDDetail()">✕</button>' +
            '</div>' +
            '<div style="border-bottom:1px solid #f1f5f9;margin-bottom:12px;padding-bottom:12px">' +
                '<div class="jd-meta-row">' +
                    '<span class="jd-meta-item"><span class="jd-meta-label">🎓 学历</span><span class="jd-meta-val">' + (jd.education || "未填写") + '</span></span>' +
                    '<span class="jd-meta-item"><span class="jd-meta-label">💼 经验</span><span class="jd-meta-val">' + (jd.experience || "未填写") + '</span></span>' +
                    '<span class="jd-meta-item"><span class="jd-meta-label">📅 毕业届</span><span class="jd-meta-val">' + (jd.graduation_year || "不限") + '</span></span>' +
                '</div>' +
            '</div>' +
            '<div class="modal-section">' +
                '<h4>🔧 核心技能要求</h4>' +
                '<div class="skill-tags" style="max-height:80px;overflow:hidden" id="jdd_core">' + coreHtml + '</div>' +
            '</div>' +
            '<div class="modal-section">' +
                '<h4>✨ 加分技能</h4>' +
                '<div class="skill-tags" style="max-height:80px;overflow:hidden" id="jdd_bonus">' + bonusHtml + '</div>' +
            '</div>' +
            benefitsHtml +
            focusHtml +
            '<div class="modal-section">' +
                '<h4>📋 JD原文</h4>' +
                '<div class="jd-full-content">' + (jd.content || "暂无") + '</div>' +
            '</div>' +
            '<div class="action-row" style="margin-top:16px">' +
                '<button class="btn btn-sm btn-warning" onclick="doAIParseJD(\'' + id + '\')">🔄 重新AI解析</button>' +
                '<button class="btn btn-sm btn-outline" onclick="closeJDDetail();openJDEdit(\'' + id + '\')">✏️ 编辑</button>' +
            '</div>';
        document.getElementById("jdModal").classList.add("active");
        document.body.style.overflow = "hidden";
    };

    window.toggleSkills = function(el) {
        var skills = JSON.parse(decodeURIComponent(el.dataset.skills));
        var cls = el.dataset.cls;
        var parent = el.parentElement;
        var html = '';
        for (var i = 0; i < skills.length; i++) html += '<span class="' + cls + '">' + skills[i] + '</span>';
        parent.innerHTML = html + '<span class="' + cls + ' skill-more" onclick="event.stopPropagation();collapseSkills(this)" data-skills="' + encodeURIComponent(JSON.stringify(skills)) + '" data-cls="' + cls + '">收起</span>';
        parent.style.maxHeight = 'none';
    };

    window.collapseSkills = function(el) {
        var skills = JSON.parse(decodeURIComponent(el.dataset.skills));
        var cls = el.dataset.cls;
        var parent = el.parentElement;
        var show = Math.min(skills.length, 5);
        var html = '';
        for (var i = 0; i < show; i++) html += '<span class="' + cls + '">' + skills[i] + '</span>';
        if (skills.length > 5) {
            html += '<span class="' + cls + ' skill-more" onclick="event.stopPropagation();toggleSkills(this)" data-skills="' + encodeURIComponent(JSON.stringify(skills)) + '" data-cls="' + cls + '">+' + (skills.length - 5) + '</span>';
        }
        parent.innerHTML = html;
        parent.style.maxHeight = '80px';
    };

    window.closeJDDetail = function() {
        document.getElementById("jdModal").classList.remove("active");
        document.body.style.overflow = "";
    };

    window.doAIParseJD = async function(id) {
        var jd = jdData[id];
        if (!jd || !jd.content) { alert("JD内容为空，无法解析"); return; }
        try {
            var btn = document.querySelector("#jdModalBox .btn-warning");
            if (btn) { btn.disabled = true; btn.textContent = "⏳ 解析中..."; }
            var res = await fetch("/jd/" + id + "/parse", { method: "POST" });
            var d = await res.json();
            if (d.jd) {
                jdData[id] = d.jd;
                renderJDCards();
                openJDDetail(id);
            }
        } catch(e) { alert("AI解析失败: " + e); }
    };

    window.openJDEdit = function(id) {
        closeJDDetail();
        var jd = jdData[id];
        if (!jd) return;
        var box = document.getElementById("jdModalBox");
        box.innerHTML =
            '<div class="modal-header">' +
                '<h3>✏️ 编辑JD</h3>' +
                '<button class="modal-close" onclick="closeJDDetail()">✕</button>' +
            '</div>' +
            '<div class="form-group"><label>岗位名称</label><input class="input" id="editTitle" value="' + (jd.title||"") + '"></div>' +
            '<div class="form-group"><label>公司名称</label><input class="input" id="editCompany" value="' + (jd.company||"") + '"></div>' +
            '<div class="form-group"><label>学历要求</label><input class="input" id="editEducation" value="' + (jd.education||"") + '"></div>' +
            '<div class="form-group"><label>经验要求</label><input class="input" id="editExperience" value="' + (jd.experience||"") + '"></div>' +
            '<div class="form-group"><label>JD内容</label><textarea class="input" id="editContent" rows="8">' + (jd.content||"") + '</textarea></div>' +
            '<div class="action-row" style="margin-top:16px">' +
                '<button class="btn btn-primary" onclick="doUpdateJD(\'' + id + '\')">💾 保存</button>' +
                '<button class="btn btn-outline" onclick="closeJDDetail()">取消</button>' +
            '</div>';
        document.getElementById("jdModal").classList.add("active");
        document.body.style.overflow = "hidden";
    };

    window.doUpdateJD = async function(id) {
        var title = document.getElementById("editTitle").value.trim();
        var company = document.getElementById("editCompany").value.trim();
        var education = document.getElementById("editEducation").value.trim();
        var experience = document.getElementById("editExperience").value.trim();
        var content = document.getElementById("editContent").value.trim();
        try {
            var res = await fetch("/jd/" + id, {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({title:title, company:company, education:education, experience:experience, content:content})
            });
            var d = await res.json();
            alert("保存成功！");
            closeJDDetail();
            await loadJDs();
        } catch(e) { alert("保存失败: " + e); }
    };

    window.doDeleteJD = async function(id) {
        var jd = jdData[id];
        if (!confirm("确定删除 " + (jd ? (jd.title || "该JD") : id) + " 吗？")) return;
        try {
            await fetch("/jd/" + id, { method: "DELETE" });
            await loadJDs();
        } catch(e) { alert("删除失败: " + e); }
    };

    // ===== V2 简历解析与智能匹配 =====
    var resumeData = {};

    async function loadResumes() {
        try {
            var res = await fetch("/resume/list");
            resumeData = await res.json();
            renderDCTable("resume");
            refreshMatchV2Selects();
        } catch(e) { console.error("加载简历数据失败:", e); }
    }

    window.parseResume = async function(coll) {
        try {
            var btn = event.target;
            btn.disabled = true; btn.textContent = "⏳ 解析中...";
            var res = await fetch("/resume/parse", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({pdf_collection: coll})
            });
            var d = await res.json();
            if (d.resume) {
                alert("简历解析成功！已提取 " + (d.resume.skills ? d.resume.skills.length : 0) + " 个技能");
                await loadResumes();
            } else {
                alert("解析失败: " + (d.error || "未知错误"));
            }
        } catch(e) { alert("解析失败: " + e); }
    };

    window.parseAllResumes = async function() {
        var ids = Object.keys(pdfData);
        if (ids.length === 0) { alert("没有可解析的PDF"); return; }
        if (!confirm("将为所有 " + ids.length + " 个PDF进行AI解析，耗时较长，确认？")) return;
        var btn = event.target;
        btn.disabled = true; btn.textContent = "⏳ 批量解析中...";
        var done = 0;
        for (var i = 0; i < ids.length; i++) {
            btn.textContent = "⏳ 解析 " + (done+1) + "/" + ids.length + "...";
            try {
                await fetch("/resume/parse", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({pdf_collection: ids[i]})
                });
                done++;
            } catch(e) {}
        }
        alert("批量解析完成：" + done + "/" + ids.length + " 个成功");
        await loadResumes();
        btn.disabled = false; btn.textContent = "🔍 批量AI解析简历";
    };

    function refreshMatchV2Selects() {
        var rids = Object.keys(resumeData);
        var jids = Object.keys(jdData);
        var rSel = document.getElementById("matchResumeSel");
        var jSel = document.getElementById("matchJdSel");
        if (rSel) {
            rSel.innerHTML = '<option value="">-- 请选择 --</option>';
            for (var i = 0; i < rids.length; i++) {
                var r = resumeData[rids[i]];
                var skills = r.skills ? " (" + r.skills.length + "技能)" : "";
                var opt = document.createElement("option");
                opt.value = rids[i];
                opt.textContent = (r.filename || rids[i]) + skills;
                rSel.appendChild(opt);
            }
        }
        if (jSel) {
            jSel.innerHTML = '<option value="">-- 请选择 --</option>';
            for (var j = 0; j < jids.length; j++) {
                var jd = jdData[jids[j]];
                var opt = document.createElement("option");
                opt.value = jids[j];
                opt.textContent = (jd.title || jd.filename || jids[j]);
                jSel.appendChild(opt);
            }
        }
    }

    window.doMatchV2 = async function() {
        var rid = document.getElementById("matchResumeSel").value;
        var jid = document.getElementById("matchJdSel").value;
        if (!rid) { alert("请选择已解析的简历"); return; }
        if (!jid) { alert("请选择JD"); return; }
        document.getElementById("matchEmpty").style.display = "none";
        var resultEl = document.getElementById("matchResult");
        resultEl.style.display = "block";
        resultEl.innerHTML = '<div class="loading">⏳ 结构化匹配中...</div>';
        try {
            var res = await fetch("/match_v2", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({resume_id: rid, jd_id: jid})
            });
            var d = await res.json();
            if (d.error) { resultEl.innerHTML = '<div style="color:#dc2626">' + d.error + '</div>'; return; }
            renderMatchV2Result(d);
        } catch(e) {
            resultEl.innerHTML = '<div style="color:#dc2626">匹配失败: ' + e + '</div>';
        }
    };

    function renderMatchV2Result(d) {
        var scoreClass = d.score >= 70 ? "high" : (d.score >= 40 ? "mid" : "low");
        var skillBarsHtml = "";
        if (d.skill_coverage) {
            for (var s = 0; s < d.skill_coverage.length; s++) {
                var sc = d.skill_coverage[s];
                var isHit = sc.matched;
                var tagType = sc.type === "required" ? '<span class="skill-tag-req">必需</span>' : '<span class="skill-tag-pref">加分</span>';
                skillBarsHtml +=
                    '<div class="skill-bar-row">' +
                        '<span class="skill-name" title="' + sc.skill + '">' + sc.skill + '</span>' +
                        tagType +
                        '<div class="skill-bar-track"><div class="skill-bar-fill ' + (isHit ? 'hit' : 'miss') + '" style="width:' + (isHit ? '100' : '0') + '%"></div></div>' +
                        '<span class="skill-status ' + (isHit ? 'hit' : 'miss') + '">' + (isHit ? '✅ 命中' : '❌ 缺失') + '</span>' +
                    '</div>';
            }
        }
        // 命中技能标签
        var matchedTagsHtml = "";
        if (d.matched_required) {
            for (var m = 0; m < d.matched_required.length; m++) {
                matchedTagsHtml += '<span class="tag-hit">✅ ' + d.matched_required[m].jd_skill + '</span>';
            }
        }
        if (d.matched_preferred) {
            for (var p = 0; p < d.matched_preferred.length; p++) {
                matchedTagsHtml += '<span class="tag-pref-hit">✨ ' + d.matched_preferred[p].jd_skill + '</span>';
            }
        }
        // 缺失技能标签
        var missingTagsHtml = "";
        if (d.missing_required) {
            for (var mr = 0; mr < d.missing_required.length; mr++) {
                missingTagsHtml += '<span class="tag-miss">❌ ' + d.missing_required[mr] + '</span>';
            }
        }
        if (d.missing_preferred) {
            for (var mp = 0; mp < d.missing_preferred.length; mp++) {
                missingTagsHtml += '<span class="tag-miss">⚠ ' + d.missing_preferred[mp] + '</span>';
            }
        }
        // 项目分析
        var projHtml = "";
        if (d.matched_projects && d.matched_projects.length > 0) {
            projHtml = '<table><thead><tr><th>项目</th><th>技术栈</th><th style="width:80px">相关度</th></tr></thead><tbody>';
            for (var pi = 0; pi < d.matched_projects.length; pi++) {
                var proj = d.matched_projects[pi];
                projHtml += '<tr><td>' + proj.name + '</td><td style="font-size:11px;color:#64748b">' + (proj.tech_stack||[]).join(", ") + '</td><td><span class="badge ' + (proj.relevance >= 0.5 ? 'badge-success' : 'badge-warn') + '">' + Math.round(proj.relevance*100) + '%</span></td></tr>';
            }
            projHtml += '</tbody></table>';
        }
        // 风险项
        var risksHtml = "";
        if (d.risks) {
            for (var r = 0; r < d.risks.length; r++) {
                risksHtml += '<div class="risk-item">⚠ ' + d.risks[r] + '</div>';
            }
        }
        document.getElementById("matchResult").innerHTML =
            '<div class="match-score-gauge">' +
                '<div class="big-score ' + scoreClass + '">' + d.score + '</div>' +
                '<div class="score-label">综合匹配度 (满分100)</div>' +
                '<div class="score-bars">' +
                    '<div class="score-bar-item"><div class="bar-label">技能 50%</div><div class="bar-track"><div class="bar-fill" style="width:' + d.skill_score + '%;background:#0ea5e9"></div></div><div class="bar-pct">' + d.skill_score + '%</div></div>' +
                    '<div class="score-bar-item"><div class="bar-label">项目 25%</div><div class="bar-track"><div class="bar-fill" style="width:' + d.project_score + '%;background:#10b981"></div></div><div class="bar-pct">' + d.project_score + '%</div></div>' +
                    '<div class="score-bar-item"><div class="bar-label">学历 10%</div><div class="bar-track"><div class="bar-fill" style="width:' + d.edu_score + '%;background:#f59e0b"></div></div><div class="bar-pct">' + d.edu_score + '%</div></div>' +
                    '<div class="score-bar-item"><div class="bar-label">经验 15%</div><div class="bar-track"><div class="bar-fill" style="width:' + d.exp_score + '%;background:#8b5cf6"></div></div><div class="bar-pct">' + d.exp_score + '%</div></div>' +
                '</div>' +
                '<div style="margin-top:12px;font-size:13px;font-weight:600">🎯 录用概率：<span style="color:#38bdf8">' + d.hire_probability + '</span></div>' +
            '</div>' +
            '<div class="card" style="margin-top:12px">' +
                '<h4>📊 技能覆盖图</h4>' +
                '<div class="skill-coverage">' + (skillBarsHtml || '<div style="font-size:12px;color:#94a3b8">JD未定义技能要求</div>') + '</div>' +
            '</div>' +
            '<div class="card">' +
                '<h4>✅ 已命中技能 (' + ((d.matched_required||[]).length + (d.matched_preferred||[]).length) + ')</h4>' +
                '<div class="match-tags">' + (matchedTagsHtml || '<span style="font-size:12px;color:#94a3b8">无</span>') + '</div>' +
            '</div>' +
            '<div class="card">' +
                '<h4>❌ 缺失技能 (' + ((d.missing_required||[]).length + (d.missing_preferred||[]).length) + ')</h4>' +
                '<div class="match-tags">' + (missingTagsHtml || '<span style="font-size:12px;color:#94a3b8">无</span>') + '</div>' +
            '</div>' +
            '<div class="card">' +
                '<h4>📋 项目匹配 (' + (d.resume_projects_count || 0) + '个项目)</h4>' +
                '<div class="table-wrap">' + (projHtml || '<div style="font-size:12px;color:#94a3b8">暂无项目经历</div>') + '</div>' +
            '</div>' +
            '<div class="card">' +
                '<h4>⚠ 风险项</h4>' +
                risksHtml +
            '</div>';
    };

    // ===== V4 面试聊天 (SaaS风格) =====

    function numToStars(v) {
        if (!v || v < 20) return "☆☆☆☆☆";
        if (v < 40) return "★☆☆☆☆";
        if (v < 55) return "★★☆☆☆";
        if (v < 70) return "★★★☆☆";
        if (v < 85) return "★★★★☆";
        return "★★★★★";
    }

    function updateDashboardStars(scores) {
        if (!scores) return;
        document.getElementById("ivStarsExpr").textContent = numToStars(scores.expression);
        document.getElementById("ivStarsProj").textContent = numToStars(scores.project_authenticity);
        document.getElementById("ivStarsMatch").textContent = numToStars(scores.job_match);
        var overall = scores.overall || 0;
        var perf = document.getElementById("ivDashPerf");
        if (perf && overall > 0) {
            var dot = perf.querySelector(".iv-perf-dot");
            var txt = perf.querySelector("span:last-child");
            if (dot) {
                if (overall >= 80) dot.style.background = "#10b981";
                else if (overall >= 60) dot.style.background = "#0ea5e9";
                else if (overall >= 40) dot.style.background = "#f59e0b";
                else dot.style.background = "#ef4444";
            }
            if (txt) {
                if (overall >= 80) txt.textContent = "表现优秀";
                else if (overall >= 60) txt.textContent = "表现良好";
                else if (overall >= 40) txt.textContent = "需要加油";
                else txt.textContent = "继续努力";
            }
        }
    }
    var ivSessionId = null;
    var ivCurrentQ = "";
    var ivState = "idle";
    var ivMode = "standard";
    var ivMatchScore = 0;
    var ivCapabilityTree = {};
    var ivCandidateProfile = {};
    var ivGap = {};

    window.selectMode = function(mode) {
        ivMode = mode;
        var cards = document.querySelectorAll(".mode-card");
        for (var i = 0; i < cards.length; i++) cards[i].classList.remove("selected");
        var card = document.querySelector('.mode-card[data-mode="' + mode + '"]');
        if (card) card.classList.add("selected");
    };

    function refreshIVSelects() {
        var rids = Object.keys(resumeData);
        if (rids.length === 0) rids = Object.keys(pdfData);
        var rSel = document.getElementById("ivResumeSel");
        if (rSel) {
            var curVal = rSel.value;
            rSel.innerHTML = '<option value="">-- 请选择 --</option>';
            for (var i = 0; i < rids.length; i++) {
                var r = resumeData[rids[i]] || pdfData[rids[i]];
                var fname = r.filename || rids[i];
                var parsed = resumeData[rids[i]] ? " (已解析)" : "";
                var opt = document.createElement("option");
                opt.value = rids[i];
                opt.textContent = fname + parsed;
                rSel.appendChild(opt);
            }
            rSel.value = curVal || "";
            rSel.onchange = function() {
                var rid = this.value;
                var r = resumeData[rid];
                var preview = document.getElementById("ivResumePreview");
                if (r) {
                    preview.innerHTML = '技能: ' + (r.skills ? r.skills.slice(0,5).join(', ') : '--') + '<br>项目: ' + (r.projects ? r.projects.length : 0) + '个';
                } else { preview.innerHTML = ''; }
            };
        }
        var jids = Object.keys(jdData);
        var jSel = document.getElementById("ivJdSel");
        if (jSel) {
            var curJVal = jSel.value;
            jSel.innerHTML = '<option value="">-- 请选择 --</option>';
            for (var j = 0; j < jids.length; j++) {
                var jd = jdData[jids[j]];
                var opt = document.createElement("option");
                opt.value = jids[j];
                opt.textContent = (jd.title || jd.filename || jids[j]);
                jSel.appendChild(opt);
            }
            jSel.value = curJVal || "";
            jSel.onchange = function() {
                var jid = this.value;
                var jd = jdData[jid];
                var preview = document.getElementById("ivJdPreview");
                if (jd) {
                    preview.innerHTML = '公司: ' + (jd.company||'--') + '<br>技能: ' + (jd.core_skills ? jd.core_skills.slice(0,5).join(', ') : '--');
                } else { preview.innerHTML = ''; }
            };
        }
    }

    window.startInterview = async function() {
        var rid = document.getElementById("ivResumeSel").value;
        var jid = document.getElementById("ivJdSel").value;
        if (!rid) { alert("请选择简历"); return; }
        if (!jid) { alert("请选择JD"); return; }
        if (!resumeData[rid]) { alert("该简历尚未AI解析，请先在资料中心上传并解析"); return; }
        if (!ivMode) { alert("请选择面试模式"); return; }
        document.getElementById("ivStartBtn").disabled = true;
        document.getElementById("ivStartBtn").textContent = "⏳ 分析中...";
        try {
            var res = await fetch("/interview/start", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({resume_id:rid,jd_id:jid,interview_mode:ivMode,interview_type:"comprehensive"})});
            var d = await res.json();
            if (d.error) { alert(d.error); document.getElementById("ivStartBtn").disabled = false; document.getElementById("ivStartBtn").textContent = "🚀 开始面试"; return; }
            ivSessionId = d.session_id;
            ivCurrentQ = d.question;
            ivState = "active";
            ivMatchScore = d.match_score || 0;
            ivCapabilityTree = d.capability_tree || {};
            ivCandidateProfile = d.candidate_profile || {};
            ivGap = d.gap || {};
            document.getElementById("interview-create").style.display = "none";
            document.getElementById("interview-active").style.display = "block";
            document.getElementById("interview-report").style.display = "none";
            document.getElementById("ivChatBody").innerHTML = '';
            addChatMsg("ai", d.question, (d.topic || "") + " · " + (d.difficulty || ""));
            document.getElementById("ivAnswerInput").value = "";
            document.getElementById("ivAnswerInput").focus();
            resetScores();
            // 顶部状态栏
            var resumeInfo = resumeData[rid];
            var jdInfo = jdData[jid];
            document.getElementById("ivTopTitle").textContent = "🎯 " + (jdInfo ? (jdInfo.title || jdInfo.filename || "--") : "--");
            document.getElementById("ivTopSub").textContent = "📄 " + (resumeInfo ? (resumeInfo.filename || "--") : "--");
            document.getElementById("ivTopMode").textContent = d.mode || "校招标准";
            document.getElementById("ivProgCur").textContent = "1";
            document.getElementById("ivProgMax").textContent = d.total_rounds || 6;
            document.getElementById("ivProgPct").textContent = Math.round(1 / (d.total_rounds || 6) * 100) + "%";
            document.getElementById("ivProgFill").style.width = Math.round(1 / (d.total_rounds || 6) * 100) + "%";
            document.getElementById("ivProgEta").textContent = "预计剩余 " + ((d.total_rounds || 6) * 3) + " 分钟";
            // 仪表盘
            document.getElementById("ivStarsExpr").textContent = "☆☆☆☆☆";
            document.getElementById("ivStarsProj").textContent = "☆☆☆☆☆";
            document.getElementById("ivStarsMatch").textContent = "☆☆☆☆☆";
            document.getElementById("ivDashPerf").querySelector("span:last-child").textContent = "--";
            document.getElementById("ivDashPerf").querySelector(".iv-perf-dot").style.background = "#94a3b8";
            // 能力差距初始化仪表盘
            if (ivGap.strengths && ivGap.strengths.length > 0) {
                document.getElementById("ivDashStrengthsSec").style.display = "block";
                document.getElementById("ivDashStrengths").innerHTML = ivGap.strengths.map(function(s){ return "<li>" + s + "</li>"; }).join("");
            }
            if (ivGap.weaknesses && ivGap.weaknesses.length > 0) {
                document.getElementById("ivDashRisksSec").style.display = "block";
                document.getElementById("ivDashRisks").innerHTML = ivGap.weaknesses.map(function(w){ return "<li>" + w + "</li>"; }).join("");
            }
        } catch(e) { alert("启动面试失败: " + e); }
        document.getElementById("ivStartBtn").disabled = false;
        document.getElementById("ivStartBtn").textContent = "🚀 开始面试";
    };

    window.submitAnswer = async function() {
        if (ivState !== "active") return;
        var answer = document.getElementById("ivAnswerInput").value.trim();
        if (!answer) return;
        document.getElementById("ivSendBtn").disabled = true;
        addChatMsg("you", answer);
        document.getElementById("ivAnswerInput").value = "";
        try {
            var res = await fetch("/interview/answer", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:ivSessionId,question:ivCurrentQ,answer:answer})});
            var d = await res.json();
            if (d.error) { alert(d.error); return; }
            if (d.evaluation) addChatEval(d.evaluation);
            if (d.quality) updateQualityScores(d.quality);
            if (d.accumulated) updateScores(d.accumulated);
            if (d.ended) {
                ivState = "completed";
                addChatMsg("ai", "📋 " + (d.message || "面试结束，正在生成报告..."));
                document.getElementById("ivProgFill").style.width = "100%";
                document.getElementById("ivProgPct").textContent = "100%";
                await doGenerateReport();
            } else if (d.type === "followup") {
                ivCurrentQ = d.question;
                var fuLabel = "追问" + (d.fu_count||"?") + "/" + (d.max_fu||4);
                if (d.topic) fuLabel += " · " + d.topic;
                if (d.difficulty) fuLabel += " · " + d.difficulty;
                addChatMsg("ai", d.question, fuLabel);
            } else if (d.type === "next") {
                ivCurrentQ = d.question;
                var nLabel = "";
                if (d.topic) nLabel += d.topic;
                if (d.difficulty) nLabel += (nLabel ? " · " : "") + d.difficulty;
                addChatMsg("ai", d.question, nLabel);
                // 更新进度条
                var max = parseInt(document.getElementById("ivProgMax").textContent) || 6;
                var cur = d.round || 1;
                document.getElementById("ivProgCur").textContent = cur;
                var pct = Math.round(cur / max * 100);
                document.getElementById("ivProgPct").textContent = pct + "%";
                document.getElementById("ivProgFill").style.width = pct + "%";
                document.getElementById("ivProgEta").textContent = "预计剩余 " + ((max - cur) * 3) + " 分钟";
            }
        } catch(e) { addChatEval("错误: " + e); }
        document.getElementById("ivSendBtn").disabled = false;
        document.getElementById("ivAnswerInput").focus();
    };

    window.endInterview = async function() {
        if (!confirm("确定结束面试吗？将生成面试报告。")) return;
        document.getElementById("ivSendBtn").disabled = true;
        try {
            var res = await fetch("/interview/end", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:ivSessionId,question:ivCurrentQ,answer:document.getElementById("ivAnswerInput").value.trim()||""})});
            var d = await res.json();
            ivState = "completed";
            if (d.report) showReport(d.report, d.final_scores);
        } catch(e) { alert("结束失败: " + e); }
        document.getElementById("ivSendBtn").disabled = false;
    };

    async function doGenerateReport() {
        var res = await fetch("/interview/end", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:ivSessionId,question:ivCurrentQ,answer:""})});
        var d = await res.json();
        if (d.report) showReport(d.report, d.final_scores);
    }

    function showReport(report, scores) {
        document.getElementById("interview-active").style.display = "none";
        document.getElementById("interview-report").style.display = "block";
        // 修复按钮
        var reportDiv = document.getElementById("interview-report");
        var existBtns = reportDiv.querySelectorAll("button");
        existBtns.forEach(function(b){ if (b.textContent.indexOf("开始新") >= 0) b.remove(); });
        var btn = document.createElement("button");
        btn.className = "btn btn-primary btn-lg";
        btn.textContent = "🔄 开始新的面试";
        btn.style.cssText = "display:block;max-width:780px;margin:12px auto 0;width:100%";
        btn.onclick = function(){ resetInterview(); };
        reportDiv.appendChild(btn);
        // 顶部大数字
        document.getElementById("ivRptScore").textContent = (scores && scores.overall) ? scores.overall : "--";
        document.getElementById("ivRptMatch").textContent = ivMatchScore ? ivMatchScore + "%" : "--";
        // 星级展示
        var dims = [
            {name:"表达", key:"expression"},
            {name:"逻辑", key:"logic"},
            {name:"项目", key:"project_authenticity"},
            {name:"技术", key:"technical"},
            {name:"抗压", key:"stress_resistance"},
            {name:"匹配", key:"job_match"}
        ];
        var starsHtml = "";
        for (var i = 0; i < dims.length; i++) {
            var v = scores ? (scores[dims[i].key] || 0) : 0;
            starsHtml += '<div style="text-align:center;min-width:60px"><div style="font-size:16px;color:#f59e0b">' + numToStars(v) + '</div><div style="font-size:10px;color:#94a3b8">' + dims[i].name + '</div></div>';
        }
        document.getElementById("ivRptStars").innerHTML = starsHtml;
        // 能力画像
        var contentEl = document.getElementById("ivReportContent");
        var profileHtml = "";
        if (ivCandidateProfile && Object.keys(ivCandidateProfile).length > 0) {
            profileHtml = '<div style="margin-bottom:16px"><div style="font-size:11px;font-weight:700;color:#64748b;margin-bottom:8px">能力画像</div><div style="display:flex;flex-wrap:wrap;gap:6px">';
            var items = Object.entries(ivCandidateProfile).sort(function(a,b){return b[1]-a[1];}).slice(0,10);
            for (var p = 0; p < items.length; p++) {
                var color = items[p][1] >= 70 ? '#10b981' : items[p][1] >= 50 ? '#f59e0b' : '#94a3b8';
                profileHtml += '<span style="font-size:11px;padding:3px 10px;border-radius:14px;background:#f8fafc;border:1px solid #edf0f5">' +
                    items[p][0] + ' <b style="color:' + color + '">' + items[p][1] + '</b></span>';
            }
            profileHtml += '</div></div>';
        }
        contentEl.innerHTML = profileHtml + '<div class="md-output">' + renderMd(report || "") + '</div>';
    }

    window.resetInterview = function() {
        ivState = "idle"; ivSessionId = null; ivMode = "standard";
        ivMatchScore = 0; ivCapabilityTree = {}; ivCandidateProfile = {}; ivGap = {};
        document.getElementById("interview-create").style.display = "block";
        document.getElementById("interview-active").style.display = "none";
        document.getElementById("interview-report").style.display = "none";
        selectMode("standard");
        refreshIVSelects();
    };

    // --- 面试记录中心 V3 ---
    window.showHistory = async function() {
        try {
            var res = await fetch("/interview/history");
            var items = await res.json();
            if (!items || items.length === 0) { 
                var box0 = document.getElementById("jdModalBox");
                box0.innerHTML = '<div class="modal-header"><h3>📂 面试记录</h3><button class="modal-close" onclick="closeJDDetail()">✕</button></div><div class="empty-state" style="padding:40px"><div class="empty-icon">📭</div><div>暂无面试记录</div><p style="font-size:12px;color:#94a3b8;margin-top:8px">完成一次模拟面试后，记录会显示在这里</p></div>';
                document.getElementById("jdModal").classList.add("active");
                document.body.style.overflow = "hidden";
                return; 
            }
            var modeIcons = {"intern":"🎓","standard":"📚","bigtech":"🏢","pressure":"🔥"};
            var html = "";
            for (var i = 0; i < items.length; i++) {
                var it = items[i];
                var statusColor = it.status === "completed" ? "#10b981" : "#f59e0b";
                var icon = (it.mode && modeIcons[it.mode]) ? modeIcons[it.mode] : "🎯";
                html += '<div class="iv-history-card" onclick="openReplay(\''+ it.id +'\')">' +
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start">' +
                        '<div>' +
                            '<div style="font-size:14px;font-weight:700;color:#0f172a">'+icon+' '+(it.jd_title||"未命名")+'</div>' +
                            '<div style="font-size:11px;color:#94a3b8;margin-top:2px">📄 '+(it.resume_name||"--")+' · 🏢 '+(it.jd_company||"--")+'</div>' +
                            '<div style="font-size:10px;color:#cbd5e1;margin-top:2px">' + (it.created_at?it.created_at.substring(0,16):"--") + ' · ' + (it.mode||"") + ' · ' + (it.rounds_count||0) + '轮</div>' +
                        '</div>' +
                        '<div style="text-align:center;min-width:60px">' +
                            '<div style="font-size:24px;font-weight:800;color:#0ea5e9">' + (it.score||"--") + '</div>' +
                            '<div style="font-size:10px;color:#94a3b8">分</div>' +
                            '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+statusColor+';margin-top:4px"></span> ' + 
                            '<span style="font-size:10px;color:#94a3b8">' + (it.status==="completed"?"完成":"进行中") + '</span>' +
                        '</div>' +
                    '</div>' +
                    '<div style="margin-top:10px;display:flex;gap:8px">' +
                        '<button class="btn btn-sm btn-outline" onclick="event.stopPropagation();openReplay(\''+it.id+'\')">📋 查看复盘</button>' +
                        '<button class="btn btn-sm btn-outline" onclick="event.stopPropagation();openReplay(\''+it.id+'\')">📊 报告</button>' +
                    '</div>' +
                '</div>';
            }
            var box = document.getElementById("jdModalBox");
            box.innerHTML = '<div class="modal-header"><h3>📂 面试记录 (' + items.length + ')</h3><button class="modal-close" onclick="closeJDDetail()">✕</button></div><div style="max-height:65vh;overflow-y:auto;padding:4px">' + html + '</div>';
            document.getElementById("jdModal").classList.add("active");
            document.body.style.overflow = "hidden";
        } catch(e) { alert("加载失败: "+e); }
    };

    // ===== 面试复盘页 =====
    window.openReplay = async function(sid) {
        try {
            var res = await fetch("/interview/report/" + sid);
            var d = await res.json();
            if (d.error) { alert(d.error); return; }
            closeJDDetail();
            document.getElementById("interview-create").style.display = "none";
            document.getElementById("interview-active").style.display = "none";
            document.getElementById("interview-report").style.display = "block";

            var scores = d.final_scores || {};
            var rounds = d.rounds || [];
            
            // 构建弱项/强项分析
            var dimNames = {expression:"表达",logic:"逻辑",project_authenticity:"项目",job_match:"匹配",technical:"技术",stress_resistance:"抗压"};
            var scoredDims = [];
            for (var k in dimNames) scoredDims.push({name: dimNames[k], val: scores[k] || 0});
            scoredDims.sort(function(a,b){return b.val - a.val;});
            var top3 = scoredDims.slice(0,3);
            var bottom3 = scoredDims.slice(-3).reverse();
            
            // 概览卡片
            var grade = scores.overall >= 80 ? "优秀" : scores.overall >= 60 ? "良好" : scores.overall >= 40 ? "一般" : "需提升";
            var topHtml = '<div class="card" style="text-align:center;padding:28px 20px">' +
                '<div style="font-size:40px;font-weight:800;color:#0ea5e9">' + (scores.overall||"--") + '</div>' +
                '<div style="font-size:12px;color:#94a3b8;margin-bottom:2px">综合评分</div>' +
                '<div style="display:flex;gap:24px;justify-content:center;margin-top:14px;flex-wrap:wrap">' +
                    '<div><div style="font-size:18px;font-weight:700;color:#334155">' + (d.match_score||"--") + '%</div><div style="font-size:10px;color:#94a3b8">岗位匹配</div></div>' +
                    '<div><div style="font-size:18px;font-weight:700;color:#334155">' + grade + '</div><div style="font-size:10px;color:#94a3b8">表现评级</div></div>' +
                    '<div><div style="font-size:18px;font-weight:700;color:#334155">' + (d.mode||"--") + '</div><div style="font-size:10px;color:#94a3b8">面试模式</div></div>' +
                '</div>' +
                '<div style="display:flex;gap:12px;justify-content:center;margin-top:16px;flex-wrap:wrap">' +
                    '<div style="background:#f0fdf4;border-radius:12px;padding:10px 16px;text-align:left;min-width:140px">' +
                        '<div style="font-size:10px;font-weight:700;color:#166534;margin-bottom:6px">✅ 最强3项</div>' +
                        top3.map(function(dim){return '<div style="font-size:11px;color:#334155;margin-bottom:2px">'+dim.name+' <b>'+dim.val+'</b></div>';}).join("") +
                    '</div>' +
                    '<div style="background:#fff7ed;border-radius:12px;padding:10px 16px;text-align:left;min-width:140px">' +
                        '<div style="font-size:10px;font-weight:700;color:#c2410c;margin-bottom:6px">⚠ 最弱3项</div>' +
                        bottom3.map(function(dim){return '<div style="font-size:11px;color:#334155;margin-bottom:2px">'+dim.name+' <b>'+dim.val+'</b></div>';}).join("") +
                    '</div>' +
                '</div></div>';

            // 时间轴
            var answered = rounds.filter(function(r){return r.answer;});
            var timelineHtml = '<div class="card" style="margin-top:12px"><h4>📝 面试时间轴 (' + answered.length + ' 题)</h4><div class="iv-replay-timeline">';
            for (var q = 0; q < rounds.length; q++) {
                var rd = rounds[q];
                if (!rd.answer) continue;
                var qType = rd.type === "followup" ? "追问" : "Q" + rd.round;
                var qScore = rd.scores || {};
                var qOverall = 0, qCnt = 0;
                for (var sk in qScore) { qOverall += qScore[sk]; qCnt++; }
                qOverall = qCnt > 0 ? Math.round(qOverall/qCnt) : 0;
                var qQuality = rd.quality || {};
                var qualDetail = "";
                if (Object.keys(qQuality).length > 0) {
                    qualDetail = (qQuality.completeness||0)+' · '+(qQuality.accuracy||0)+' · '+(qQuality.depth||0)+' · '+(qQuality.authenticity||0);
                }
                var sc = qOverall >= 80 ? "#10b981" : qOverall >= 60 ? "#0ea5e9" : qOverall >= 40 ? "#f59e0b" : "#ef4444";
                
                timelineHtml += '<div class="iv-replay-item" id="replayQ' + q + '">' +
                    '<div class="iv-replay-round">' + qType + (rd.difficulty ? ' · ' + rd.difficulty : '') + (rd.topic ? ' · ' + rd.topic : '') + '</div>' +
                    '<div class="iv-replay-bubble ai-bubble"><b>AI 提问：</b>' + rd.question + '</div>' +
                    '<div class="iv-replay-bubble you-bubble"><b>我的回答：</b>' + (rd.answer || "(未回答)") + '</div>';
                
                if (rd.evaluation) {
                    timelineHtml += '<div class="iv-replay-eval"><b>AI 点评：</b>' + rd.evaluation;
                    if (qualDetail) timelineHtml += '<div style="font-size:10px;color:#94a3b8;margin-top:3px">完整度·准确度·深度·真实性: ' + qualDetail + '</div>';
                    timelineHtml += '</div>';
                }
                
                timelineHtml += '<div style="display:flex;align-items:center;gap:10px;margin:6px 0 8px 20px">' +
                    '<div style="font-size:20px;font-weight:800;color:' + sc + '">' + qOverall + '</div>' +
                    '<div style="font-size:11px;color:#94a3b8">本题得分</div>' +
                    '<button class="btn btn-sm btn-outline" onclick="genBestAnswer(\'' + sid + '\',' + q + ')" style="margin-left:auto;font-size:11px">💡 查看优秀答案</button>' +
                    '</div>' +
                    '<div id="bestAnswer' + q + '" style="display:none;margin:6px 0 6px 20px;padding:10px 14px;background:#f0fdf4;border-radius:10px;border-left:3px solid #10b981;font-size:12px;color:#334155;line-height:1.65"></div>' +
                    '</div>';
            }
            timelineHtml += '</div></div>';

            // 能力画像
            var profile = d.candidate_profile || {};
            if (Object.keys(profile).length > 0) {
                timelineHtml += '<div class="card" style="margin-top:12px"><h4>🎯 能力画像</h4><div style="display:flex;flex-wrap:wrap;gap:6px">';
                var items = Object.entries(profile).sort(function(a,b){return b[1]-a[1];}).slice(0,12);
                for (var p = 0; p < items.length; p++) {
                    var color = items[p][1] >= 70 ? '#10b981' : items[p][1] >= 50 ? '#f59e0b' : '#94a3b8';
                    timelineHtml += '<span style="font-size:11px;padding:3px 10px;border-radius:14px;background:#f8fafc;border:1px solid #edf0f5">' +
                        items[p][0] + ' <b style="color:' + color + '">' + items[p][1] + '</b></span>';
                }
                timelineHtml += '</div></div>';
            }

            document.getElementById("ivReportContent").innerHTML = topHtml + timelineHtml;
            var reportDiv = document.getElementById("interview-report");
            var existBtns = reportDiv.querySelectorAll("button");
            if (existBtns.length === 0) {
                var btn = document.createElement("button");
                btn.className = "btn btn-primary btn-lg";
                btn.textContent = "🔄 开始新的面试";
                btn.style.cssText = "display:block;max-width:780px;margin:12px auto 0;width:100%";
                btn.onclick = function(){ resetInterview(); };
                reportDiv.appendChild(btn);
            }
            reportDiv.scrollIntoView({behavior:"smooth"});
        } catch(e) { alert("加载复盘失败: " + e); }
    };

    // ===== 生成最佳答案 =====
    window.genBestAnswer = async function(sid, roundIdx) {
        var el = document.getElementById("bestAnswer" + roundIdx);
        if (!el) return;
        if (el.style.display === "block") { el.style.display = "none"; return; }
        el.style.display = "block";
        el.textContent = "⏳ 正在生成优秀答案...";
        try {
            var res = await fetch("/interview/best_answer", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:sid,round_index:roundIdx})});
            var d = await res.json();
            if (d.error) { el.textContent = "生成失败: " + d.error; return; }
            el.innerHTML = '<div style="font-size:10px;font-weight:700;color:#166534;margin-bottom:4px">💡 参考答案</div>' + 
                '<div>' + d.best_answer.replace(/\n/g, "<br>") + '</div>' +
                '<div style="font-size:10px;color:#94a3b8;margin-top:8px;border-top:1px solid #d1fae5;padding-top:6px">你的回答：' + (d.user_answer||"").substring(0,150) + '...</div>';
        } catch(e) { el.textContent = "生成失败: " + e; }
    };

    function addChatMsg(role, text, label) {
        var body = document.getElementById("ivChatBody");
        if (!body) return;
        var div = document.createElement("div");
        div.className = "iv-msg " + (role === "ai" ? "msg-ai" : "msg-you");
        
        // Avatar
        var avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = role === "ai" ? "AI" : "U";
        div.appendChild(avatar);
        
        var bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        
        // 标签
        if (label && role === "ai") {
            var parts = label.split(" · ");
            if (parts.length > 0) {
                var tagDiv = document.createElement("div");
                tagDiv.className = "msg-tags";
                for (var p = 0; p < parts.length; p++) {
                    var t = parts[p].trim();
                    if (!t) continue;
                    var tag = document.createElement("span");
                    tag.className = "msg-tag";
                    if (t.indexOf("追问") >= 0) tag.classList.add("tag-followup");
                    else if (t.indexOf("项目") >= 0 || t.indexOf("深挖") >= 0) tag.classList.add("tag-deep");
                    else if (t.indexOf("技术") >= 0 || t.indexOf("基础") >= 0) tag.classList.add("tag-tech");
                    else if (t.indexOf("能力") >= 0 || t.indexOf("岗位") >= 0 || t.indexOf("加分") >= 0) tag.classList.add("tag-ability");
                    else if (t.indexOf("行为") >= 0 || t.indexOf("开放") >= 0) tag.classList.add("tag-behavior");
                    else tag.classList.add("tag-basic");
                    tag.textContent = t;
                    tagDiv.appendChild(tag);
                }
                bubble.appendChild(tagDiv);
            }
        }
        
        var textEl = document.createElement("div");
        textEl.className = "msg-text";
        textEl.textContent = text;
        bubble.appendChild(textEl);
        div.appendChild(bubble);
        
        body.appendChild(div);
        body.scrollTop = body.scrollHeight;
    }

    function addChatEval(text) {
        var body = document.getElementById("ivChatBody");
        if (!body) return;
        var last = body.lastElementChild;
        if (last && last.classList.contains("iv-msg") && last.classList.contains("msg-you")) {
            var evalEl = document.createElement("div");
            evalEl.className = "msg-eval";
            evalEl.textContent = "💬 " + text;
            last.querySelector(".msg-bubble").appendChild(evalEl);
        }
        body.scrollTop = body.scrollHeight;
    }

    var ivDimMap = {"expression":"Expr","logic":"Logic","project_authenticity":"Proj","job_match":"Match","technical":"Tech","stress_resistance":"Stress"};

    function updateQualityScores(quality) {
        // V4: 注入到AI消息后方的eval气泡，不显示raw数字
        if (!quality) return;
        var avg = Math.round((quality.completeness + quality.accuracy + quality.depth + quality.authenticity) / 4);
        var stars = numToStars(avg);
        var evalText = "表现良好";
        if (avg >= 80) evalText = "回答优秀";
        else if (avg >= 60) evalText = "表现良好";
        else if (avg >= 40) evalText = "可以更好";
        else evalText = "需要改进";
        var body = document.getElementById("ivChatBody");
        if (!body) return;
        var msgs = body.querySelectorAll(".iv-msg");
        var lastYou = null;
        for (var i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].classList.contains("msg-you")) { lastYou = msgs[i]; break; }
        }
        if (lastYou) {
            var nextAi = lastYou.nextElementSibling;
            if (nextAi && nextAi.classList.contains("msg-ai") && !nextAi.querySelector(".msg-eval")) {
                var evalEl = document.createElement("div");
                evalEl.className = "msg-eval";
                evalEl.innerHTML = '<div class="eval-stars">' + stars + ' ' + evalText + '</div>' +
                    '<div class="eval-pros">完整度 ' + (quality.completeness||0) + ' · 准确度 ' + (quality.accuracy||0) + '</div>' +
                    '<div class="eval-cons">深度 ' + (quality.depth||0) + ' · 真实性 ' + (quality.authenticity||0) + '</div>';
                nextAi.querySelector(".msg-bubble").appendChild(evalEl);
            }
        }
    }

    function updateScores(accum, label) {
        if (!accum) return;
        var overall = 0, cnt = 0;
        for (var k in ivDimMap) { if (accum[k] !== undefined) { overall += accum[k]; cnt++; } }
        accum.overall = cnt > 0 ? Math.round(overall / cnt) : 0;
        updateDashboardStars(accum);
    }

    function resetScores() {
        updateScores({expression:0,logic:0,project_authenticity:0,job_match:0,technical:0,stress_resistance:0}, "--");
        document.getElementById("ivProgCur").textContent = "1";
        document.getElementById("ivProgPct").textContent = "0%";
        document.getElementById("ivProgFill").style.width = "0%";
        document.getElementById("ivProgEta").textContent = "预计剩余 -- 分钟";
    }

    // 初始化时加载简历数据
    var _origOnload = window.onload;
    window.onload = function() {
        if (_origOnload) _origOnload();
        loadResumes();
    };
    // ===== 图片导入 =====
    window.openImportModal = function() {
        var box = document.getElementById("jdModalBox");
        box.innerHTML =
            '<div class="modal-header"><h3>📥 导入资料</h3><button class="modal-close" onclick="closeJDDetail()">✕</button></div>' +
            '<div class="modal-section"><h4>📎 选择文件</h4>' +
                '<input type="file" id="importFiles" class="input w-full" accept=".png,.jpg,.jpeg,.webp,.bmp,.pdf" multiple style="padding:10px">' +
                '<div style="font-size:11px;color:#94a3b8;margin-top:4px">支持 PNG/JPG/WEBP/BMP/PDF，可多选。PDF直接导入，图片会OCR识别。</div>' +
            '</div>' +
            '<div class="modal-section">' +
                '<h4>📋 导入类型 <span id="importAutoDetect" style="font-size:11px;color:#10b981;font-weight:400"></span></h4>' +
                '<label style="display:flex;align-items:center;gap:8px;margin-bottom:6px;cursor:pointer"><input type="radio" name="importType" value="jd" checked> 💼 作为JD导入</label>' +
                '<label style="display:flex;align-items:center;gap:8px;cursor:pointer"><input type="radio" name="importType" value="resume"> 📄 作为简历导入</label>' +
            '</div>' +
            '<div id="importPreview" style="display:none" class="modal-section">' +
                '<h4>📝 OCR识别预览</h4>' +
                '<div id="importPreviewText" style="background:#f8fafc;border-radius:8px;padding:12px;font-size:12px;color:#475569;line-height:1.6;white-space:pre-wrap;max-height:300px;overflow-y:auto;border:1px solid #e2e8f0"></div>' +
                '<div style="font-size:11px;color:#94a3b8;margin-top:4px">请检查识别结果，确认无误后导入</div>' +
            '</div>' +
            '<div class="action-row" style="margin-top:16px">' +
                '<button class="btn btn-primary" onclick="runImportOCR()">🔍 识别预览</button>' +
                '<button class="btn btn-success" onclick="confirmImport()" id="importConfirmBtn" style="display:none">✅ 确认导入</button>' +
            '</div>';
        document.getElementById("jdModal").classList.add("active");
        document.body.style.overflow = "hidden";
    };

    window.runImportOCR = async function() {
        var files = document.getElementById("importFiles").files;
        if (!files || files.length === 0) { alert("请选择文件"); return; }
        var btn = event.target;
        btn.disabled = true; btn.textContent = "⏳ 识别中...";

        var fd = new FormData();
        for (var i = 0; i < files.length; i++) {
            fd.append("files", files[i]);
        }
        try {
            var res = await fetch("/import/ocr", { method: "POST", body: fd });
            var d = await res.json();
            if (d.error) { alert(d.error); btn.disabled = false; btn.textContent = "🔍 识别预览"; return; }
            document.getElementById("importPreview").style.display = "block";
            document.getElementById("importPreviewText").textContent = d.text || "(无文本)";
            document.getElementById("importConfirmBtn").style.display = "inline-flex";
            window._importOcrText = d.text;
            // 自动选择检测类型
            if (d.detected_type) {
                var radios = document.querySelectorAll('input[name="importType"]');
                for (var ri = 0; ri < radios.length; ri++) {
                    radios[ri].checked = (radios[ri].value === d.detected_type);
                }
                var autoEl = document.getElementById("importAutoDetect");
                if (autoEl) autoEl.textContent = "(自动识别: " + (d.detected_type === "jd" ? "JD" : "简历") + (d.ocr_engine ? " · " + d.ocr_engine : "") + ")";
            }
        } catch(e) { alert("识别失败: " + e); }
        btn.disabled = false; btn.textContent = "🔍 识别预览";
    };

    window.confirmImport = async function() {
        var text = window._importOcrText;
        if (!text || !text.trim()) { alert("识别文本为空"); return; }
        var typeEl = document.querySelector('input[name="importType"]:checked');
        var type = typeEl ? typeEl.value : "jd";
        var btn = document.getElementById("importConfirmBtn");
        btn.disabled = true; btn.textContent = "⏳ 导入中...";
        try {
            var res = await fetch("/import/confirm", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ocr_text: text, import_type: type})
            });
            var d = await res.json();
            if (d.error) { alert(d.error); btn.disabled = false; btn.textContent = "✅ 确认导入"; return; }
            alert(type === "jd" ? "JD导入成功！" : "简历导入成功！");
            closeJDDetail();
            if (type === "jd") await loadJDs();
            else await refreshPDFData();
        } catch(e) { alert("导入失败: " + e); }
        btn.disabled = false; btn.textContent = "✅ 确认导入";
    };
    </script>
    </body>
    </html>
    """

pdf_registry = {}
pdf_cache = {}
bm25_cache = {}  # {collection_name: BM25Okapi对象}

# ====== 统一资料中心 V2 ======
DOCUMENT_REGISTRY_FILE = "document_registry.json"
document_registry = {}   # {doc_id: metadata}
document_cache = {}      # {doc_id: [chunk strings]}

def load_document_registry():
    global document_registry
    try:
        with open(DOCUMENT_REGISTRY_FILE, "r", encoding="utf-8") as f:
            document_registry = json.load(f)
    except:
        document_registry = {}

def save_document_registry():
    with open(DOCUMENT_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(document_registry, f, ensure_ascii=False, indent=2)

load_document_registry()

def build_bm25(collection_name, chunks):
    """为指定知识库构建BM25索引（jieba中文分词）"""
    tokenized = [list(jieba.cut(chunk)) for chunk in chunks if chunk.strip()]
    if tokenized:
        bm25_cache[collection_name] = BM25Okapi(tokenized)

# 👉 启动时加载历史PDF记录
try:
    with open("pdf_registry.json", "r", encoding="utf-8") as f:
        pdf_registry = json.load(f)
except:
    pdf_registry = {}

# 👉 启动时重建pdf_cache和bm25_cache + 迁移到document_registry
for cid, info in pdf_registry.items():
    try:
        file_path = f"uploads/{info['filename']}"
        text = ""
        if os.path.exists(file_path):
            suffix = info['filename'].rsplit(".", 1)[-1].lower() if "." in info['filename'] else "pdf"
            if suffix == "pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
        if not text:
            continue
        chunk_size = 500
        overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        pdf_cache[cid] = chunks
        build_bm25(cid, chunks)
        # 迁移到统一document_registry
        if cid not in document_registry:
            doc_id = f"doc_{cid}"
            document_registry[doc_id] = {
                "filename": info["filename"],
                "type": "knowledge",
                "source_type": suffix if suffix in ("pdf","txt","md") else "image",
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "chunk_count": len(chunks),
                "file_path": file_path,
                "text_length": len(text),
                "collection_name": cid
            }
            document_cache[doc_id] = chunks
            save_document_registry()
    except:
        pass

# 👉 迁移JD registry中基于文件的条目到document_registry
try:
    with open("jd_registry.json", "r", encoding="utf-8") as f:
        jd_reg = json.load(f)
    for jid, jd_info in jd_reg.items():
        fname = jd_info.get("filename")
        if fname and fname != "图片导入":
            doc_id = f"doc_{jid}"
            if doc_id not in document_registry:
                doc_type = "jd"
                src = jd_info.get("source", "pdf")
                document_registry[doc_id] = {
                    "filename": fname,
                    "type": doc_type,
                    "source_type": src if src in ("pdf","image","text") else "pdf",
                    "upload_time": jd_info.get("created_at", "")[:19].replace("T"," "),
                    "chunk_count": 0,
                    "file_path": f"uploads/{fname}",
                    "text_length": len(jd_info.get("content","")),
                    "jd_id": jid
                }
                save_document_registry()
except: pass

# 👉 迁移Resume registry中基于文件的条目到document_registry
try:
    with open("resume_registry.json", "r", encoding="utf-8") as f:
        res_reg = json.load(f)
    for rid, res_info in res_reg.items():
        fname = res_info.get("filename")
        cid = res_info.get("pdf_collection")
        if cid:
            doc_id = f"doc_{cid}"
            if doc_id not in document_registry:
                document_registry[doc_id] = {
                    "filename": fname or "未知简历",
                    "type": "resume",
                    "source_type": "pdf",
                    "upload_time": res_info.get("parsed_at", "")[:19].replace("T"," "),
                    "chunk_count": pdf_registry.get(cid, {}).get("chunks", 0),
                    "file_path": f"uploads/{fname}" if fname else "",
                    "text_length": len(res_info.get("raw_text","")),
                    "collection_name": cid
                }
                save_document_registry()
except: pass

# ====== 统一资料中心 API ======

def extract_text_from_file(file_path: str, suffix: str) -> tuple:
    """统一文本提取，返回 (text, source_type, metadata_dict)"""
    import time as _time
    meta = {}
    
    if suffix == "pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            pt = page.extract_text()
            if pt:
                text += pt + "\n"
        return text, "pdf", meta
    
    elif suffix in ("png", "jpg", "jpeg", "webp", "bmp"):
        t0 = _time.time()
        try:
            text = run_ocr(file_path)
            meta["ocr_engine"] = _ocr_mode
        except Exception as e:
            raise RuntimeError(f"图片识别失败: {str(e)}")
        meta["ocr_time_ms"] = round((_time.time() - t0) * 1000)
        return text, "image", meta
    
    elif suffix == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, "text", meta
    
    elif suffix == "md":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, "text", meta
    
    else:
        raise ValueError(f"不支持的文件类型: {suffix}，支持 PDF/PNG/JPG/JPEG/WEBP/DOCX/TXT/MD")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """统一分块"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_and_store_chunks(doc_id: str, chunks: list):
    """将chunks向量化并存入ChromaDB"""
    import time as _time
    t0 = _time.time()
    count = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"collection": doc_id}]
        )
        count += 1
    embed_time_ms = round((_time.time() - t0) * 1000)
    return count, embed_time_ms

# ===== 统一上传接口 =====
@app.post("/documents/upload")
async def documents_upload(
    file: UploadFile = File(...),
    document_type: str = "knowledge"
):
    """
    统一文件上传
    - file: 文件
    - document_type: "knowledge" | "jd" | "resume"
    """
    import time as _time
    fname = file.filename or "unknown"
    suffix = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    file_size = 0
    
    print(f"[统一上传] 文件: {fname}, 类型: {suffix}, 资料类型: {document_type}")
    
    # 保存文件
    file_path = f"uploads/{fname}"
    try:
        content = await file.read()
        file_size = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        return {"success": False, "message": f"文件保存失败: {str(e)}"}
    
    # 提取文本
    try:
        text, source_type, extract_meta = extract_text_from_file(file_path, suffix)
    except ValueError as e:
        return {"success": False, "message": str(e)}
    except RuntimeError as e:
        # OCR失败容错
        print(f"[统一上传] OCR失败: {e}")
        return {"success": False, "message": str(e)}
    
    text = text.strip()
    if not text:
        return {"success": False, "message": "文件内容为空，请检查文件是否正确"}
    
    text_length = len(text)
    print(f"[统一上传] 文本长度: {text_length}, 来源类型: {source_type}")
    
    # 分块
    chunks = chunk_text(text)
    chunk_count = len(chunks)
    print(f"[统一上传] Chunk数量: {chunk_count}")
    
    # 生成ID
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    # 向量化
    try:
        embed_count, embed_time_ms = embed_and_store_chunks(doc_id, chunks)
        print(f"[统一上传] 向量化完成: {embed_count}条, 耗时: {embed_time_ms}ms")
    except Exception as e:
        return {"success": False, "message": f"向量化失败: {str(e)}"}
    
    # 写入registry
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    document_registry[doc_id] = {
        "filename": fname,
        "type": document_type,
        "source_type": source_type,
        "upload_time": now_str,
        "chunk_count": chunk_count,
        "text_length": text_length,
        "file_path": file_path,
        "file_size": file_size,
        "embed_count": embed_count,
        "ocr_engine": extract_meta.get("ocr_engine", ""),
        "ocr_time_ms": extract_meta.get("ocr_time_ms", 0),
        "embed_time_ms": embed_time_ms,
        "collection_name": doc_id
    }
    save_document_registry()
    
    # 写入缓存（兼容旧逻辑）
    document_cache[doc_id] = chunks
    pdf_cache[doc_id] = chunks  # 兼容旧代码
    build_bm25(doc_id, chunks)
    
    # 同时写入旧的pdf_registry以保持兼容
    pdf_registry[doc_id] = {
        "filename": fname,
        "chunks": chunk_count
    }
    with open("pdf_registry.json", "w", encoding="utf-8") as f:
        json.dump(pdf_registry, f, ensure_ascii=False, indent=4)
    
    # 向后兼容：JD和简历类型自动写入旧registry
    if document_type == "jd":
        jd_id = f"jd_{uuid.uuid4().hex[:8]}"
        jd_now = datetime.now().isoformat()
        jd_registry[jd_id] = {
            "id": jd_id, "title": fname.rsplit(".", 1)[0] if "." in fname else fname,
            "company": "", "content": text,
            "core_skills": [], "bonus_skills": [], "benefits": [],
            "education": "", "experience": "", "graduation_year": None,
            "interview_focus": [], "filename": fname, "source": source_type,
            "quality_score": compute_jd_quality({"content": text}),
            "created_at": jd_now, "updated_at": jd_now,
            "doc_id": doc_id
        }
        save_jd_registry()
        # 自动AI解析JD
        try:
            parsed = parse_jd_with_ai(text)
            jd_registry[jd_id].update({
                "title": parsed.get("title") or jd_registry[jd_id]["title"],
                "company": parsed.get("company") or "",
                "core_skills": parsed.get("core_skills", []),
                "bonus_skills": parsed.get("bonus_skills", []),
                "benefits": parsed.get("benefits", []),
                "education": parsed.get("education", ""),
                "experience": parsed.get("experience", ""),
                "graduation_year": parsed.get("graduation_year"),
                "interview_focus": parsed.get("interview_focus", []),
                "updated_at": datetime.now().isoformat()
            })
            jd_registry[jd_id]["quality_score"] = compute_jd_quality(jd_registry[jd_id])
            save_jd_registry()
        except: pass
    elif document_type == "resume":
        try:
            parsed = parse_resume_with_ai(text)
            resume_id = f"res_{doc_id}"
            resume_registry[resume_id] = {
                "id": resume_id, "pdf_collection": doc_id,
                "filename": fname, "skills": parsed.get("skills", []),
                "projects": parsed.get("projects", []),
                "education": parsed.get("education", {}),
                "internships": parsed.get("internships", []),
                "certificates": parsed.get("certificates", []),
                "total_years": parsed.get("total_years", "0"),
                "summary": parsed.get("summary", ""),
                "raw_text": text, "parsed_at": datetime.now().isoformat(),
                "doc_id": doc_id
            }
            save_resume_registry()
        except: pass
    
    print(f"[统一上传] ✅ 完成: {fname} → {doc_id} | 文本{text_length}字 | {chunk_count}chunks | {embed_count}向量 | OCR:{extract_meta.get('ocr_time_ms',0)}ms | Embed:{embed_time_ms}ms")
    
    return {
        "success": True,
        "message": "上传成功",
        "document": {
            "id": doc_id,
            "filename": fname,
            "type": document_type,
            "source_type": source_type,
            "chunk_count": chunk_count,
            "text_length": text_length,
            "upload_time": now_str
        }
    }

# ===== 资料列表接口 =====
@app.get("/documents")
def documents_list(type: str = ""):
    """
    获取资料列表，支持过滤
    - type: ""(全部) | "knowledge" | "jd" | "resume"
    """
    result = {}
    for doc_id, info in document_registry.items():
        if type and info.get("type") != type:
            continue
        result[doc_id] = {
            "id": doc_id,
            "filename": info.get("filename", ""),
            "type": info.get("type", "knowledge"),
            "source_type": info.get("source_type", "pdf"),
            "upload_time": info.get("upload_time", ""),
            "chunk_count": info.get("chunk_count", 0),
            "text_length": info.get("text_length", 0),
            "file_size": info.get("file_size", 0)
        }
    return result

# ===== 资料详情接口 =====
@app.get("/documents/{doc_id}")
def documents_detail(doc_id: str):
    """获取单个资料详情，包含chunk预览"""
    if doc_id not in document_registry:
        return {"error": "资料不存在"}
    
    info = document_registry[doc_id]
    chunks = document_cache.get(doc_id, [])
    
    chunk_previews = []
    for i, chunk in enumerate(chunks):
        chunk_previews.append({
            "index": i + 1,
            "preview": chunk[:100] + ("..." if len(chunk) > 100 else ""),
            "length": len(chunk),
            "full_text": chunk
        })
    
    return {
        "id": doc_id,
        "filename": info.get("filename", ""),
        "type": info.get("type", "knowledge"),
        "source_type": info.get("source_type", "pdf"),
        "upload_time": info.get("upload_time", ""),
        "chunk_count": info.get("chunk_count", 0),
        "text_length": info.get("text_length", 0),
        "file_size": info.get("file_size", 0),
        "ocr_engine": info.get("ocr_engine", ""),
        "ocr_time_ms": info.get("ocr_time_ms", 0),
        "embed_time_ms": info.get("embed_time_ms", 0),
        "chunks": chunk_previews
    }

# ===== 删除资料接口 =====
@app.delete("/documents/{doc_id}")
def documents_delete(doc_id: str):
    """删除资料（registry + 向量 + 缓存）"""
    if doc_id not in document_registry:
        return {"error": "资料不存在"}
    
    info = document_registry[doc_id]
    fname = info.get("filename", "")
    
    # 删除向量库数据
    try:
        collection.delete(where={"collection": doc_id})
    except Exception:
        pass
    
    # 删除缓存
    if doc_id in document_cache:
        del document_cache[doc_id]
    if doc_id in pdf_cache:
        del pdf_cache[doc_id]
    if doc_id in bm25_cache:
        del bm25_cache[doc_id]
    
    # 删除registry
    del document_registry[doc_id]
    save_document_registry()
    
    # 同步删除旧pdf_registry
    if doc_id in pdf_registry:
        del pdf_registry[doc_id]
        with open("pdf_registry.json", "w", encoding="utf-8") as f:
            json.dump(pdf_registry, f, ensure_ascii=False, indent=4)
    
    print(f"[统一删除] {doc_id} ({fname}) 已删除")
    return {"message": "删除成功", "filename": fname}

# ===== 兼容旧接口（内部委托给统一系统） =====
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """[兼容] 旧版上传，统一委托给 /documents/upload"""
    return await documents_upload(file=file, document_type="knowledge")

@app.get("/pdfs")
def get_pdfs():
    """[兼容] 返回旧格式pdf_registry + 新格式"""
    return pdf_registry

@app.delete("/pdf/{collection_name}")
def delete_pdf(collection_name: str):
    """[兼容] 旧版删除，尝试统一删除"""
    # 先尝试统一删除
    if collection_name in document_registry:
        return documents_delete(collection_name)
    
    # 旧版删除逻辑
    if collection_name in pdf_registry:
        del pdf_registry[collection_name]
        with open("pdf_registry.json", "w", encoding="utf-8") as f:
            json.dump(pdf_registry, f, ensure_ascii=False, indent=4)
    try:
        collection.delete(where={"collection": collection_name})
    except:
        pass
    if collection_name in pdf_cache:
        del pdf_cache[collection_name]
    if collection_name in bm25_cache:
        del bm25_cache[collection_name]
    return {"message": "删除成功"}

# ====== 知识库统计 ======

@app.get("/stats")
def get_stats():
    total_chunks = sum(info.get("chunk_count", 0) for info in document_registry.values())
    vector_count = collection.count()
    
    # 按类型统计
    kb_count = sum(1 for i in document_registry.values() if i.get("type") == "knowledge")
    jd_count = sum(1 for i in document_registry.values() if i.get("type") == "jd")
    resume_count = sum(1 for i in document_registry.values() if i.get("type") == "resume")
    
    pdf_files = [
        {
            "collection": cid,
            "filename": info["filename"],
            "chunks": info.get("chunk_count", info.get("chunks", 0))
        }
        for cid, info in pdf_registry.items()
    ]

    return {
        "pdf_count": len(document_registry),
        "chunk_count": total_chunks,
        "vector_count": vector_count,
        "kb_count": kb_count,
        "jd_count": jd_count,
        "resume_count": resume_count,
        "files": pdf_files
    }

# ====== AI接口 ======

@app.post("/chat")
def chat(data: ChatRequest):

    # 校验：至少选择一个PDF
    if not data.collection_names:
        return {
            "answer": "请至少勾选一个PDF。",
            "sources": []
        }

    # ======================
    # summary模式 —— 多PDF汇总
    # ======================
    if data.mode == "summary":

        results = collection.get(
            where={
                "collection": {"$in": data.collection_names}
            }
        )

        if not results["documents"]:
            return {
                "answer": "没有找到所选PDF的内容。",
                "sources": []
            }

        context = "\n".join(results["documents"])

        # 构建来源信息
        sources = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            coll_name = meta.get("collection", "")
            filename = pdf_registry.get(coll_name, {}).get("filename", "未知文件")
            sources.append({
                "filename": filename,
                "collection": coll_name,
                "content": doc[:300] + ("..." if len(doc) > 300 else ""),
                "distance": None
            })

    # ======================
    # QA模式 —— 升级版检索链路
    # 阶段一: 向量Top20 + BM25 Top20 → 融合去重
    # 阶段二: CrossEncoder重排序 → Top5
    # ======================
    else:

        question_embedding = embedding_model.encode(data.question)
        query_tokens = list(jieba.cut(data.question))
        candidate_pool = {}  # key: (collection, content_hash) → match_info

        for cname in data.collection_names:
            # ── 阶段一: 向量检索 Top20 ──
            vec_results = collection.query(
                query_embeddings=[question_embedding.tolist()],
                n_results=20,
                where={"collection": cname}
            )
            if vec_results["documents"][0]:
                for i, doc in enumerate(vec_results["documents"][0]):
                    dist = vec_results["distances"][0][i]
                    # 真实距离 → 相似度: ChromaDB L2, sim = max(0, 1 - dist/2) * 100
                    vec_sim = round(max(0, 1 - dist / 2) * 100, 1)
                    key = cname + "|" + doc[:60]
                    filename = pdf_registry.get(cname, {}).get("filename", "未知文件")
                    candidate_pool[key] = {
                        "filename": filename, "collection": cname, "content": doc,
                        "distance": round(dist, 4), "vector_score": vec_sim,
                        "bm25_score": 0.0, "hybrid_score": 0.0, "cross_score": 0.0
                    }

            # ── 阶段一: BM25 检索 Top20 ──
            if cname in bm25_cache:
                bm25 = bm25_cache[cname]
                bm25_scores = bm25.get_scores(query_tokens)
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
                chunks = pdf_cache.get(cname, [])
                bm25_top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:20]
                for idx in bm25_top_indices:
                    if not chunks[idx].strip(): continue
                    bm25_norm = round(bm25_scores[idx] / max_bm25 * 100, 1)
                    chunk_text = chunks[idx]
                    key = cname + "|" + chunk_text[:60]
                    filename = pdf_registry.get(cname, {}).get("filename", "未知文件")
                    if key in candidate_pool:
                        candidate_pool[key]["bm25_score"] = bm25_norm
                    else:
                        candidate_pool[key] = {
                            "filename": filename, "collection": cname, "content": chunk_text,
                            "distance": None, "vector_score": 0.0,
                            "bm25_score": bm25_norm, "hybrid_score": 0.0, "cross_score": 0.0
                        }

        # ── 融合: hybrid_score = vec*0.5 + bm25*0.5 ──
        candidates = list(candidate_pool.values())
        for c in candidates:
            c["hybrid_score"] = round(c["vector_score"] * 0.5 + c["bm25_score"] * 0.5, 1)

        if not candidates:
            return {"answer": "没有找到相关内容。", "sources": []}

        # 按hybrid降序取 Top20 送入 CrossEncoder
        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
        rerank_pool = candidates[:20]

        # ── 阶段二: CrossEncoder 重排序 ──
        try:
            pairs = [[data.question, c["content"]] for c in rerank_pool]
            cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            cross_scores = [float(s) for s in cross_scores]  # numpy→Python float
            # 归一化 CrossEncoder scores 到 0-100
            min_cs = min(cross_scores); max_cs = max(cross_scores)
            if max_cs > min_cs:
                for i, c in enumerate(rerank_pool):
                    c["cross_score"] = round((cross_scores[i] - min_cs) / (max_cs - min_cs) * 100, 1)
            else:
                for c in rerank_pool:
                    c["cross_score"] = 50.0
            rerank_pool.sort(key=lambda c: c["cross_score"], reverse=True)
        except Exception:
            # CrossEncoder失败时降级为hybrid排序
            for c in rerank_pool:
                c["cross_score"] = c["hybrid_score"]

        # ── 最终 Top5 ──
        top_matches = rerank_pool[:5]

        # 构建 context
        context = "\n\n---\n\n".join(
            f"[来源: {m['filename']}](Cross:{m['cross_score']}%)\n{m['content']}"
            for m in top_matches
        )

        # ── 构建来源信息 ──
        sources = []
        for idx, m in enumerate(top_matches):
            prev_chunk = ""; next_chunk = ""
            cname = m["collection"]
            if cname in pdf_cache:
                chunks = pdf_cache[cname]
                target = m["content"].strip()
                for ci in range(len(chunks)):
                    if chunks[ci].strip() == target:
                        if ci > 0: prev_chunk = chunks[ci - 1].strip()
                        if ci < len(chunks) - 1: next_chunk = chunks[ci + 1].strip()
                        break
            sources.append({
                "filename": m["filename"],
                "collection": m["collection"],
                "content": m["content"],
                "prev_chunk": prev_chunk,
                "next_chunk": next_chunk,
                "distance": m["distance"],
                "vector_score": m["vector_score"],
                "bm25_score": m["bm25_score"],
                "hybrid_score": m["hybrid_score"],
                "cross_score": m["cross_score"],
                "rank": idx + 1
            })

    # ======================
    # LLM
    # ======================
    # 告知LLM当前检索了哪些文件
    selected_files = []
    for cn in data.collection_names:
        fname = pdf_registry.get(cn, {}).get("filename", cn)
        selected_files.append(fname)

    prompt = f"""
请根据以下资料回答问题。

你正在检索以下文件：
{chr(10).join(f'- {f}' for f in selected_files)}

资料：
{context}

问题：
{data.question}

如果资料中没有答案，请明确说明。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }

# ====== JD岗位匹配分析 ======

@app.post("/job_match")
def job_match(data: JobMatchRequest):

    # 获取简历全文
    resume_docs = collection.get(
        where={"collection": data.resume_collection}
    )
    if not resume_docs["documents"]:
        return {"answer": "简历PDF内容为空。", "match_rate": None}
    resume_text = "\n".join(resume_docs["documents"])

    # 获取JD全文
    jd_docs = collection.get(
        where={"collection": data.jd_collection}
    )
    if not jd_docs["documents"]:
        return {"answer": "JD内容为空。", "match_rate": None}
    jd_text = "\n".join(jd_docs["documents"])

    resume_name = pdf_registry.get(data.resume_collection, {}).get("filename", "未知")
    jd_name = pdf_registry.get(data.jd_collection, {}).get("filename", "未知")

    prompt = f"""
你是一名资深技术面试官和HR专家。请基于以下简历和JD进行深度岗位匹配分析。

## 简历内容（{resume_name}）
{resume_text}

## 岗位JD（{jd_name}）
{jd_text}

## 分析要求

请用Markdown格式输出以下完整分析：

### 一、综合匹配度
给出百分比，如：**82%**

### 二、技能匹配项
列出JD要求且简历具备的技能，使用 ✅ 标记，每项一行。
格式：✅ Java —— 简历中项目A、项目B均使用

### 三、缺失技能
列出JD要求但简历未体现的技能，使用 ❌ 标记，每项一行。
格式：❌ Redis —— JD要求缓存经验，简历未提及

### 四、项目经验匹配分析
- 项目数量是否匹配JD级别要求
- 项目复杂度评估
- 与岗位关联度分析

### 五、简历竞争力分析
- 核心优势（3~5点）
- 风险点（2~3点）
- 简历优化建议（具体可操作）

### 六、技能补齐计划
- 优先学习清单（按重要性排序，Top 5）
- 建议学习周期（紧急/短期/长期）

### 七、面试通过率预测
给出百分比并解释原因。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "resume": resume_name,
        "jd": jd_name
    }

# ====== JD管理模块 ======

JD_REGISTRY_FILE = "jd_registry.json"

class JDCreateRequest(BaseModel):
    title: str = ""
    company: str = ""
    content: str

class JDUpdateRequest(BaseModel):
    title: str = ""
    company: str = ""
    content: str = ""
    education: str = ""
    experience: str = ""

# 加载JD registry
try:
    with open(JD_REGISTRY_FILE, "r", encoding="utf-8") as f:
        jd_registry = json.load(f)
except:
    jd_registry = {}

def save_jd_registry():
    with open(JD_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(jd_registry, f, ensure_ascii=False, indent=2)

def parse_jd_with_ai(content: str) -> dict:
    """调用LLM提取JD结构化信息"""
    prompt = f"""你是资深HR专家。请解析以下JD内容，提取结构化信息并以JSON格式返回。

JD内容：
{content[:3000]}

请返回以下JSON（只返回JSON，不要markdown代码块标记）：
{{
  "title": "岗位名称",
  "company": "公司名称（如无法识别填'未知'）",
  "core_skills": ["核心技能1", "核心技能2"],
  "bonus_skills": ["加分技能（候选人能力要求，如：有AI经验优先、熟悉K8s优先）"],
  "benefits": ["企业福利（公司待遇，如：实习可转正、导师带教、餐补、住宿补贴。严格要求：不要将福利放入bonus_skills）"],
  "education": "学历要求",
  "experience": "经验要求（注意：如果是应届生/202X届/校招，经验应为'应届生/无经验要求'，不要将毕业年份当作工作经验）",
  "graduation_year": "毕业年份（如2027，应届毕业生填当前年份，无则填null）",
  "interview_focus": ["面试重点1", "面试重点2"]
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    # 尝试提取JSON
    parsed = None
    try:
        parsed = json.loads(raw)
    except:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                parsed = json.loads(match.group())
            except:
                pass
    if parsed is None:
        parsed = {
            "title": "", "company": "",
            "core_skills": [], "bonus_skills": [],
            "benefits": [], "education": "", "experience": "",
            "graduation_year": None,
            "interview_focus": []
        }

    # 后处理：将误归入bonus_skills的企业福利迁移到benefits
    benefit_keywords = ["转正","导师","餐补","补贴","下午茶","健身房","团建","五险一金","公积金","社保","年假","双休","弹性","班车","免费","零食","饮料","水果","体检","年终奖","股票","期权","旅游","生日","节日","礼金","住房","落户","人才","交通","通讯","加班","调休"]
    if parsed and "bonus_skills" in parsed:
        real_bonus = []
        migrated = []
        for s in list(parsed.get("bonus_skills", [])):
            is_benefit = any(kw in str(s) for kw in benefit_keywords)
            if is_benefit:
                migrated.append(s)
            else:
                real_bonus.append(s)
        parsed["bonus_skills"] = real_bonus
        if migrated:
            existing = parsed.get("benefits") or []
            parsed["benefits"] = existing + migrated
    if "benefits" not in parsed:
        parsed["benefits"] = []

    # 后处理：检测应届生相关词汇
    content_lower = content.lower()
    fresh_grad_patterns = [
        r'(\d{4})届', r'(\d{4})年毕业', r'应届毕业生',
        r'应届生', r'校招', r'校园招聘', r'实习生', r'实习'
    ]
    has_fresh_indicators = False
    detected_year = None
    for pattern in fresh_grad_patterns:
        m = re.search(pattern, content)
        if m:
            has_fresh_indicators = True
            if m.groups() and m.group(1) and m.group(1).isdigit():
                year = int(m.group(1))
                if 2000 <= year <= 2100:
                    detected_year = year
            break

    # 如果是应届/校招，修正 experience 和 graduation_year
    if has_fresh_indicators and detected_year:
        parsed["graduation_year"] = detected_year
        # 防止2027届被理解为2027年经验
        if parsed.get("experience") and any(c.isdigit() for c in str(parsed["experience"])):
            exp_str = str(parsed["experience"])
            if str(detected_year) in exp_str:
                parsed["experience"] = "应届生/无经验要求"
        if not parsed.get("experience") or "年" in str(parsed.get("experience", "")):
            # 检查是否是数字年+年模式
            exp_match = re.search(r'(\d{4})\s*年', str(parsed.get("experience", "")))
            if exp_match:
                parsed["experience"] = "应届生/无经验要求"
    elif has_fresh_indicators:
        parsed["graduation_year"] = datetime.now().year
        parsed["experience"] = "应届生/无经验要求"

    return parsed


def compute_jd_quality(jd: dict) -> int:
    """计算JD完整度评分 0-100"""
    score = 0
    if jd.get("title") and jd["title"] not in ("未命名JD", "", None):
        score += 15
    if jd.get("company") and jd["company"] not in ("未填写", "未知", "", None):
        score += 10
    if jd.get("core_skills") and len(jd.get("core_skills", [])) >= 3:
        score += 25
    elif jd.get("core_skills") and len(jd.get("core_skills", [])) > 0:
        score += 15
    if jd.get("bonus_skills") and len(jd.get("bonus_skills", [])) > 0:
        score += 10
    if jd.get("education") and jd["education"] not in ("", None):
        # 检查是否是明确的学历描述
        edu = str(jd["education"])
        if any(kw in edu for kw in ("本科", "硕士", "博士", "大专", "专科", "不限", "以上")):
            score += 15
        else:
            score += 5
    if jd.get("experience") and jd["experience"] not in ("", None):
        exp = str(jd["experience"])
        if any(kw in exp for kw in ("年", "应届", "无经验", "经验")):
            score += 15
        else:
            score += 5
    if jd.get("content") and len(jd.get("content", "")) > 200:
        score += 10
    elif jd.get("content") and len(jd.get("content", "")) > 50:
        score += 5
    return min(score, 100)


@app.post("/jd/create")
def jd_create(data: JDCreateRequest):
    """新建JD（文本输入），自动AI解析"""
    jd_id = f"jd_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()

    jd_registry[jd_id] = {
        "id": jd_id,
        "title": data.title or "未命名JD",
        "company": data.company or "未填写",
        "content": data.content,
        "core_skills": [],
        "bonus_skills": [],
        "benefits": [],
        "education": "",
        "experience": "",
        "graduation_year": None,
        "interview_focus": [],
        "filename": None,
        "source": "text",
        "quality_score": 0,
        "created_at": now,
        "updated_at": now
    }
    save_jd_registry()

    # 自动AI解析
    try:
        parsed = parse_jd_with_ai(data.content)
        jd_registry[jd_id].update({
            "title": parsed.get("title") or jd_registry[jd_id]["title"],
            "company": parsed.get("company") or jd_registry[jd_id]["company"],
            "core_skills": parsed.get("core_skills", []),
            "bonus_skills": parsed.get("bonus_skills", []),
            "benefits": parsed.get("benefits", []),
            "education": parsed.get("education", ""),
            "experience": parsed.get("experience", ""),
            "graduation_year": parsed.get("graduation_year"),
            "interview_focus": parsed.get("interview_focus", []),
            "updated_at": datetime.now().isoformat()
        })
        jd_registry[jd_id]["quality_score"] = compute_jd_quality(jd_registry[jd_id])
        save_jd_registry()
    except Exception as e:
        pass  # AI parse is best-effort

    return {"message": "JD创建成功", "jd": jd_registry[jd_id]}


@app.post("/jd/upload_pdf")
async def jd_upload_pdf(file: UploadFile = File(...)):
    """上传JD文件（PDF或图片），提取文本并AI解析"""
    fname = file.filename or "unknown"
    suffix = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    print(f"[JD上传] 文件: {fname}, 类型: {suffix}")

    file_path = f"uploads/{fname}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return {"message": f"文件保存失败: {str(e)}"}

    text = ""
    source = "pdf"

    # 根据文件类型提取文本
    if suffix == "pdf":
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text += pt + "\n"
        except Exception as e:
            return {"message": f"PDF读取失败: {str(e)}"}
    elif suffix in ("png", "jpg", "jpeg", "webp", "bmp"):
        source = "image"
        try:
            text = run_ocr(file_path)
            print(f"[JD上传] OCR文字长度: {len(text)}")
        except Exception as e:
            return {"message": f"图片识别失败: {str(e)}"}
    else:
        return {"message": f"不支持的文件类型: {suffix}，支持 PDF/PNG/JPG/JPEG/WEBP"}

    if not text.strip():
        return {"message": "文件内容为空，请检查文件是否正确"}

    jd_id = f"jd_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()

    jd_registry[jd_id] = {
        "id": jd_id,
        "title": fname.rsplit(".", 1)[0] if "." in fname else fname,
        "company": "",
        "content": text,
        "core_skills": [],
        "bonus_skills": [],
        "benefits": [],
        "education": "",
        "experience": "",
        "graduation_year": None,
        "interview_focus": [],
        "filename": fname,
        "source": source,
        "quality_score": 0,
        "created_at": now,
        "updated_at": now
    }
    save_jd_registry()

    # 自动AI解析
    try:
        parsed = parse_jd_with_ai(text)
        jd_registry[jd_id].update({
            "title": parsed.get("title") or jd_registry[jd_id]["title"],
            "company": parsed.get("company") or "",
            "core_skills": parsed.get("core_skills", []),
            "bonus_skills": parsed.get("bonus_skills", []),
            "benefits": parsed.get("benefits", []),
            "education": parsed.get("education", ""),
            "experience": parsed.get("experience", ""),
            "graduation_year": parsed.get("graduation_year"),
            "interview_focus": parsed.get("interview_focus", []),
            "updated_at": datetime.now().isoformat()
        })
        jd_registry[jd_id]["quality_score"] = compute_jd_quality(jd_registry[jd_id])
        save_jd_registry()
    except Exception as e:
        pass

    return {"message": "JD上传解析成功", "jd": jd_registry[jd_id]}


@app.get("/jd/list")
def jd_list():
    """获取所有JD列表"""
    return jd_registry


@app.get("/jd/{jd_id}")
def jd_detail(jd_id: str):
    """获取单个JD详情"""
    if jd_id not in jd_registry:
        return {"error": "JD不存在"}
    return jd_registry[jd_id]


@app.put("/jd/{jd_id}")
def jd_update(jd_id: str, data: JDUpdateRequest):
    """更新JD信息"""
    if jd_id not in jd_registry:
        return {"error": "JD不存在"}

    jd = jd_registry[jd_id]
    if data.title:
        jd["title"] = data.title
    if data.company:
        jd["company"] = data.company
    if data.content:
        jd["content"] = data.content
    if data.education:
        jd["education"] = data.education
    if data.experience:
        jd["experience"] = data.experience
    jd["updated_at"] = datetime.now().isoformat()
    save_jd_registry()

    return {"message": "更新成功", "jd": jd}


@app.delete("/jd/{jd_id}")
def jd_delete(jd_id: str):
    """删除JD"""
    if jd_id in jd_registry:
        del jd_registry[jd_id]
        save_jd_registry()
        return {"message": "删除成功"}
    return {"error": "JD不存在"}


@app.post("/jd/{jd_id}/parse")
def jd_parse(jd_id: str):
    """重新AI解析JD"""
    if jd_id not in jd_registry:
        return {"error": "JD不存在"}

    jd = jd_registry[jd_id]
    if not jd["content"].strip():
        return {"message": "JD内容为空"}

    try:
        parsed = parse_jd_with_ai(jd["content"])
        jd.update({
            "title": parsed.get("title") or jd["title"],
            "company": parsed.get("company") or jd["company"],
            "core_skills": parsed.get("core_skills", jd.get("core_skills", [])),
            "bonus_skills": parsed.get("bonus_skills", jd.get("bonus_skills", [])),
            "benefits": parsed.get("benefits", jd.get("benefits", [])),
            "education": parsed.get("education", jd.get("education", "")),
            "experience": parsed.get("experience", jd.get("experience", "")),
            "graduation_year": parsed.get("graduation_year", jd.get("graduation_year")),
            "interview_focus": parsed.get("interview_focus", jd.get("interview_focus", [])),
            "updated_at": datetime.now().isoformat()
        })
        jd["quality_score"] = compute_jd_quality(jd)
        save_jd_registry()
        return {"message": "AI解析完成", "jd": jd}
    except Exception as e:
        return {"message": f"解析失败: {str(e)}"}


# ====== 简历-JD匹配引擎 V2 ======

RESUME_REGISTRY_FILE = "resume_registry.json"

class ResumeParseRequest(BaseModel):
    pdf_collection: str

class MatchV2Request(BaseModel):
    resume_id: str
    jd_id: str

# 加载resume registry
try:
    with open(RESUME_REGISTRY_FILE, "r", encoding="utf-8") as f:
        resume_registry = json.load(f)
except:
    resume_registry = {}

def save_resume_registry():
    with open(RESUME_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(resume_registry, f, ensure_ascii=False, indent=2)

def normalize_skill(s: str) -> str:
    """标准化技能名称"""
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")

def skills_match(skill: str, skill_list: list[str]) -> tuple[bool, str]:
    """检查skill是否匹配skill_list中的任一技能（模糊匹配）"""
    ns = normalize_skill(skill)
    for s in skill_list:
        nt = normalize_skill(s)
        if ns == nt or ns in nt or nt in ns:
            return True, s
    return False, ""

def parse_resume_with_ai(text: str) -> dict:
    """AI解析简历为结构化数据"""
    prompt = f"""你是资深HR。请解析以下简历，提取结构化信息并以JSON格式返回。

简历内容：
{text[:4000]}

请返回以下JSON（只返回JSON，不要markdown代码块）：
{{
  "skills": ["技能1", "技能2"],
  "projects": [
    {{"name": "项目名", "description": "简述", "tech_stack": ["技术1", "技术2"]}}
  ],
  "education": {{"degree": "本科", "school": "学校名", "major": "专业"}},
  "internships": [
    {{"company": "公司", "role": "岗位", "duration": "3个月"}}
  ],
  "certificates": ["证书1"],
  "total_years": "总工作经验年数（数字）",
  "summary": "一句话总结候选人"
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {
        "skills": [], "projects": [],
        "education": {}, "internships": [],
        "certificates": [], "total_years": "0",
        "summary": ""
    }


@app.post("/resume/parse")
def resume_parse(data: ResumeParseRequest):
    """解析简历PDF为结构化数据"""
    coll = data.pdf_collection
    if coll not in pdf_registry:
        return {"error": "PDF不存在"}
    if coll not in pdf_cache:
        return {"error": "PDF缓存未加载，请重启服务"}
    
    text = "\n".join(pdf_cache[coll])
    if not text.strip():
        return {"error": "简历内容为空"}

    parsed = parse_resume_with_ai(text)
    resume_id = f"res_{coll}"
    now = datetime.now().isoformat()

    resume_registry[resume_id] = {
        "id": resume_id,
        "pdf_collection": coll,
        "filename": pdf_registry[coll]["filename"],
        "skills": parsed.get("skills", []),
        "projects": parsed.get("projects", []),
        "education": parsed.get("education", {}),
        "internships": parsed.get("internships", []),
        "certificates": parsed.get("certificates", []),
        "total_years": parsed.get("total_years", "0"),
        "summary": parsed.get("summary", ""),
        "raw_text": text,
        "parsed_at": now
    }
    save_resume_registry()
    return {"message": "简历解析成功", "resume": resume_registry[resume_id]}


@app.get("/resume/list")
def resume_list():
    """获取所有已解析简历"""
    return resume_registry


def try_extract_years(text: str) -> float:
    """从文本中提取工作年限"""
    import re as _re
    # 匹配 "3年" / "3-5年" / "3年以上" 等
    m = _re.search(r'(\d+)[-~至到]?\s*(\d+)?\s*年', text)
    if m:
        if m.group(2):
            return (float(m.group(1)) + float(m.group(2))) / 2
        return float(m.group(1))
    return 0


# ===== 匹配引擎核心算法 =====

def match_resume_to_jd(resume_data: dict, jd_data: dict) -> dict:
    """
    V2 结构化匹配引擎
    
    权重:
    - 技能匹配: 50%
    - 项目相关度: 25%
    - 学历匹配: 10%
    - 经验匹配: 15%
    """
    # ===== 1. 技能匹配 (50%) =====
    resume_skills = resume_data.get("skills", [])
    jd_required = jd_data.get("core_skills", [])
    jd_preferred = jd_data.get("bonus_skills", [])

    matched_required = []
    missing_required = []
    for skill in jd_required:
        matched, original = skills_match(skill, resume_skills)
        if matched:
            matched_required.append({"jd_skill": skill, "resume_skill": original})
        else:
            missing_required.append(skill)

    matched_preferred = []
    missing_preferred = []
    for skill in jd_preferred:
        matched, original = skills_match(skill, resume_skills)
        if matched:
            matched_preferred.append({"jd_skill": skill, "resume_skill": original})
        else:
            missing_preferred.append(skill)

    # 必需技能命中率 (权70%)
    req_total = len(jd_required)
    req_hit = len(matched_required)
    req_rate = req_hit / req_total if req_total > 0 else 1.0

    # 加分技能命中率 (权30%)
    pref_total = len(jd_preferred)
    pref_hit = len(matched_preferred)
    pref_rate = pref_hit / pref_total if pref_total > 0 else 1.0

    # 如果没有必需技能，只看加分技能
    if req_total == 0 and pref_total > 0:
        skill_score = pref_rate * 100
    elif req_total == 0 and pref_total == 0:
        skill_score = 50  # 无技能要求，默认及格
    else:
        skill_score = (req_rate * 0.70 + pref_rate * 0.30) * 100

    # ===== 2. 项目相关度 (25%) =====
    projects = resume_data.get("projects", [])
    all_jd_skills = jd_required + jd_preferred
    project_score = 0
    matched_projects = []
    if projects and all_jd_skills:
        for proj in projects:
            tech_stack = proj.get("tech_stack", [])
            matched_count = 0
            for ts in tech_stack:
                m, _ = skills_match(ts, all_jd_skills)
                if m:
                    matched_count += 1
            relevance = matched_count / len(all_jd_skills) if all_jd_skills else 0
            matched_projects.append({
                "name": proj.get("name", "未知项目"),
                "tech_stack": tech_stack,
                "matched_skills": matched_count,
                "relevance": round(relevance, 2)
            })
            project_score += relevance
        project_score = (project_score / len(projects)) * 100 if projects else 50
    elif not projects:
        project_score = 30  # 无项目经历
    else:
        project_score = 60  # JD无技能要求

    # ===== 3. 学历匹配 (10%) =====
    edu_levels = {"博士": 5, "硕士": 4, "研究生": 4, "本科": 3, "学士": 3, "大专": 2, "专科": 2, "高中": 1}
    resume_edu = resume_data.get("education", {})
    resume_degree = resume_edu.get("degree", "") if isinstance(resume_edu, dict) else ""
    jd_edu_req = jd_data.get("education", "")

    resume_level = 3  # default本科
    for key, level in edu_levels.items():
        if key in str(resume_degree):
            resume_level = max(resume_level, level)
    jd_level = 3
    for key, level in edu_levels.items():
        if key in str(jd_edu_req):
            jd_level = max(jd_level, level)

    if resume_level >= jd_level:
        edu_score = 100
    elif resume_level == jd_level - 1:
        edu_score = 50
    else:
        edu_score = 0

    # ===== 4. 经验匹配 (15%) =====
    resume_years = try_extract_years(str(resume_data.get("total_years", "0")))
    jd_exp_req = jd_data.get("experience", "")
    jd_years = try_extract_years(str(jd_exp_req)) if jd_exp_req else 0

    if jd_years == 0:
        exp_score = 80
    elif resume_years >= jd_years:
        exp_score = 100
    elif resume_years >= jd_years * 0.7:
        exp_score = 70
    elif resume_years >= jd_years * 0.5:
        exp_score = 40
    else:
        exp_score = 10

    # ===== 综合评分 =====
    total_score = round(
        skill_score * 0.50 +
        project_score * 0.25 +
        edu_score * 0.10 +
        exp_score * 0.15,
        1
    )

    # ===== 录用概率 =====
    if total_score >= 85:
        hire_prob = "极高 (85%+)"
    elif total_score >= 70:
        hire_prob = "高 (60-85%)"
    elif total_score >= 55:
        hire_prob = "中等 (40-60%)"
    elif total_score >= 40:
        hire_prob = "偏低 (20-40%)"
    else:
        hire_prob = "低 (<20%)"

    # ===== 风险项 =====
    risks = []
    if req_rate < 0.5:
        risks.append(f"必需技能匹配率仅{round(req_rate*100)}%，核心技能严重不足")
    elif req_rate < 0.8:
        risks.append(f"必需技能匹配率{round(req_rate*100)}%，部分核心技能缺失")
    if resume_level < jd_level:
        risks.append(f"学历不达标（要求{jd_edu_req}）")
    if resume_years < jd_years and jd_years > 0:
        risks.append(f"工作经验不足（要求{jd_years}年，实际约{resume_years}年）")
    if not projects:
        risks.append("缺少项目经历")
    if not risks:
        risks.append("无明显风险项")

    # ===== 构建技能覆盖数据 =====
    skill_coverage = []
    for skill in jd_required:
        matched, _ = skills_match(skill, resume_skills)
        skill_coverage.append({
            "skill": skill,
            "type": "required",
            "matched": matched
        })
    for skill in jd_preferred:
        matched, _ = skills_match(skill, resume_skills)
        skill_coverage.append({
            "skill": skill,
            "type": "preferred",
            "matched": matched
        })

    return {
        "score": total_score,
        "skill_score": round(skill_score, 1),
        "project_score": round(project_score, 1),
        "edu_score": round(edu_score, 1),
        "exp_score": round(exp_score, 1),
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
        "matched_projects": matched_projects,
        "skill_coverage": skill_coverage,
        "risks": risks,
        "hire_probability": hire_prob,
        "resume_skills_count": len(resume_skills),
        "resume_projects_count": len(projects),
        "resume_education": resume_edu,
        "resume_summary": resume_data.get("summary", ""),
        "weights": {"skills": 50, "projects": 25, "education": 10, "experience": 15}
    }


@app.post("/match_v2")
def match_v2(data: MatchV2Request):
    """V2 结构化简历-JD匹配"""
    if data.resume_id not in resume_registry:
        return {"error": "简历不存在，请先解析简历"}
    if data.jd_id not in jd_registry:
        return {"error": "JD不存在"}

    resume_data = resume_registry[data.resume_id]
    jd_data = jd_registry[data.jd_id]

    result = match_resume_to_jd(resume_data, jd_data)
    result["resume_name"] = resume_data.get("filename", "")
    result["jd_title"] = jd_data.get("title", "")
    result["jd_company"] = jd_data.get("company", "")

    return result

# ====== AI面试官 V2 ======

INTERVIEW_FILE = "interview_sessions.json"

SCORE_DIMS = ["expression", "logic", "project_authenticity", "job_match", "technical", "stress_resistance"]
SCORE_DIMS_CN = {"expression":"表达能力","logic":"逻辑能力","project_authenticity":"项目真实性","job_match":"岗位匹配度","technical":"技术能力","stress_resistance":"抗压能力"}

# 技术深挖链：每个技术关键词可以延伸的更深层问题
TECH_DRILL_CHAIN = {
    "redis": ["缓存穿透/击穿/雪崩", "Redis分布式锁", "缓存一致性(Cache Aside/Write Through)", "Redis集群方案(Cluster/Sentinel)", "Redis数据结构底层实现"],
    "mysql": ["索引优化(覆盖索引/索引下推)", "SQL慢查询分析与优化", "事务隔离级别与MVCC", "分库分表方案", "主从复制与读写分离"],
    "spring": ["IOC容器原理", "AOP实现机制", "Spring事务管理", "Spring Boot自动配置", "Spring Cloud微服务组件"],
    "java": ["JVM内存模型与GC", "并发编程(synchronized/ReentrantLock)", "集合类源码(HashMap/ConcurrentHashMap)", "设计模式实践", "异常处理最佳实践"],
    "多线程": ["线程池参数调优", "锁升级机制", "AQS原理", "CompletableFuture异步编程", "ThreadLocal内存泄漏"],
    "分布式": ["CAP理论与取舍", "分布式事务(Seata/TC)", "服务治理与限流降级", "分布式ID生成方案", "配置中心与注册中心"],
    "消息队列": ["消息可靠性保证", "顺序消息实现", "消息堆积处理", "死信队列", "Kafka vs RocketMQ选型"],
    "docker": ["Dockerfile多阶段构建", "容器网络模式", "数据卷管理", "Docker Compose编排", "容器资源限制"],
    "kubernetes": ["Pod调度策略", "Service与Ingress", "ConfigMap与Secret", "HPA弹性伸缩", "Helm包管理"],
    "网络": ["TCP三次握手与四次挥手", "HTTP/HTTPS协议", "DNS解析流程", "CDN加速原理", "负载均衡算法"],
    "数据库": ["范式与反范式设计", "慢查询优化", "连接池配置", "数据库备份策略", "数据迁移方案"],
    "前端": ["浏览器渲染流程", "跨域解决方案", "前端性能优化", "状态管理方案", "微前端架构"],
    "微服务": ["服务拆分原则", "API网关设计", "链路追踪(SkyWalking)", "配置中心(Nacos)", "服务熔断降级"],
    "算法": ["时间复杂度分析", "动态规划应用", "二叉树相关算法", "排序算法比较", "算法在实际项目中的使用"],
    "设计模式": ["单例模式实现方式", "工厂模式应用场景", "观察者模式实践", "策略模式 vs 状态模式", "项目中实际使用的设计模式"],
    "测试": ["单元测试覆盖率", "Mock技术", "集成测试策略", "性能测试(JMeter)", "自动化测试框架"],
    "python": ["GIL与多线程", "装饰器原理", "生成器与协程", "Django vs FastAPI", "Python内存管理"],
    "go": ["goroutine调度", "Channel通信", "GC优化", "Go Module依赖管理", "Go测试与性能分析"],
}

def get_deep_dive_topics(keyword: str) -> list[str]:
    """根据技术关键词获取深挖方向"""
    keyword = keyword.lower().replace(" ", "").replace("-", "")
    for key, topics in TECH_DRILL_CHAIN.items():
        if key in keyword:
            return topics
    return []

class InterviewStartRequest(BaseModel):
    resume_id: str
    jd_id: str
    interview_type: str  # "hr" | "tech" | "comprehensive" [deprecated, use interview_mode]
    interview_mode: str = "standard"  # "intern" | "standard" | "bigtech" | "pressure"

class InterviewAnswerRequest(BaseModel):
    session_id: str
    question: str
    answer: str

try:
    with open(INTERVIEW_FILE, "r", encoding="utf-8") as f:
        interview_sessions = json.load(f)
except:
    interview_sessions = {}

def save_sessions():
    with open(INTERVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(interview_sessions, f, ensure_ascii=False, indent=2)

def detect_candidate_level(resume: dict) -> dict:
    """根据简历判断候选人等级
    
    返回: {level: "A"|"B"|"C"|"D", label: str, reasoning: str, 
           ban_topics: list, question_ratio: dict}
    """
    skills = resume.get("skills", [])
    projects = resume.get("projects", [])
    internships = resume.get("internships", [])
    total_years_str = str(resume.get("total_years", "0"))
    edu = resume.get("education", {})
    degree = edu.get("degree", "") if isinstance(edu, dict) else ""
    summary = resume.get("summary", "")
    
    # 尝试从total_years提取数字
    try:
        total_years = float(total_years_str)
    except:
        total_years = 0
    
    project_count = len(projects)
    intern_count = len(internships)
    skill_count = len(skills)
    
    # 判断是否有正式工作经历
    has_fulltime = any(
        kw in summary.lower() for kw in 
        ["工作", "任职", "在职", "担任", "负责", "就职于", "工作经验", "fulltime", "full-time"]
    ) or total_years >= 1
    
    # 项目复杂度判断
    has_complex_project = False
    complex_tech = ["分布式", "微服务", "高并发", "集群", "k8s", "kubernetes", "docker", 
                    "消息队列", "redis", "mysql优化", "分库分表", "架构设计",
                    "docker", "mq", "kafka", "rabbitmq", "elasticsearch", "es"]
    for p in projects:
        tech_str = " ".join(p.get("tech_stack", [])).lower()
        desc_str = p.get("description", "").lower()
        if any(t in tech_str or t in desc_str for t in complex_tech):
            has_complex_project = True
            break
    
    # 技术深度判断
    deep_skills = sum(1 for s in skills if any(
        t in s.lower() for t in ["源码", "原理", "底层", "jvm", "gc", "线程池", "分布式",
                                 "redis集群", "mysql优化", "设计模式", "架构"]
    ))
    
    # 综合判断
    reasons = []
    
    if total_years <= 0 and project_count <= 2 and intern_count <= 2 and not has_fulltime:
        if "实习" in summary or "实习生" in summary or "intern" in summary.lower():
            level = "A"; label = "实习生"
            reasons.append("简历明确标注实习身份")
        elif project_count <= 1 and skill_count <= 5:
            level = "A"; label = "实习生"
            reasons.append("项目少、技能基础，判断为实习生级别")
        else:
            level = "B"; label = "应届生"
            reasons.append(f"有{project_count}个项目但无工作经历")
    elif total_years <= 1 and project_count <= 3 and not has_complex_project:
        if "应届" in summary or "毕业生" in summary or "fresh" in summary.lower():
            level = "B"; label = "应届生"
            reasons.append("简历标注应届毕业生")
        elif intern_count >= 1 and project_count >= 2:
            level = "B"; label = "应届生"
            reasons.append(f"{intern_count}段实习+{project_count}个项目，判断为应届生")
        else:
            level = "C"; label = "初级工程师"
            reasons.append("有少量工作经验，判断为初级工程师")
    elif total_years <= 3 and has_complex_project:
        level = "C"; label = "初级工程师"
        reasons.append(f"{total_years}年经验+复杂项目，判断为初级工程师")
    elif total_years <= 3 and deep_skills >= 2:
        level = "C"; label = "初级工程师"
        reasons.append(f"{total_years}年经验+{deep_skills}项深度技能")
    elif total_years > 3 or (has_complex_project and deep_skills >= 3):
        level = "D"; label = "中级工程师"
        reasons.append(f"{total_years}年经验+复杂项目+深度技能")
    else:
        level = "B"; label = "应届生"
        reasons.append("默认判断为应届生")
    
    # 禁止话题（A/B级限制）
    ban_topics = []
    if level == "A":
        ban_topics = ["分布式系统设计", "微服务架构", "Transformer底层实现", 
                      "Redis源码分析", "JVM底层调优", "高并发架构设计",
                      "MySQL索引底层原理(B+树源码)", "消息队列底层存储",
                      "Kubernetes调度源码", "Spring源码级问题",
                      "GC算法细节", "线程池底层实现", "Netty底层",
                      "操作系统内核", "数据库引擎实现"]
    elif level == "B":
        ban_topics = ["分布式事务方案(Seata源码)", "JVM GC源码级分析",
                      "Redis Cluster源码", "消息队列底层实现",
                      "Spring Boot自动配置源码", "操作系统内核",
                      "数据库存储引擎源码", "网络协议栈实现"]
    
    # 出题比例
    question_ratio = {
        "A": {"基础验证题": 70, "项目追问题": 30, "技术提升题": 0},
        "B": {"基础验证题": 50, "项目追问题": 40, "技术提升题": 10},
        "C": {"基础验证题": 30, "项目追问题": 30, "技术提升题": 40},
        "D": {"基础验证题": 10, "项目追问题": 30, "技术提升题": 60},
    }[level]
    
    return {
        "level": level,
        "label": label,
        "reasoning": "；".join(reasons),
        "ban_topics": ban_topics,
        "question_ratio": question_ratio,
        "project_count": project_count,
        "skill_count": skill_count,
        "has_fulltime": has_fulltime,
        "has_complex_project": has_complex_project,
        "total_years": total_years
    }

# ===== V3 面试分析引擎 =====

def build_capability_tree(jd_content: str) -> dict:
    """从JD文本构建岗位能力树
    
    返回: {"技术域": ["具体能力1", "具体能力2", ...], ...}
    """
    prompt = f"""你是资深技术架构师。请分析以下JD，提取该岗位需要的能力树。

JD内容：
{jd_content[:3000]}

请分析并输出JSON（仅JSON，不要markdown标记）：
{{
  "技术域1": ["具体能力1", "具体能力2", ...],
  "技术域2": ["具体能力1", "具体能力2", ...],
  ...
}}

规则：
1. 技术域以核心技术或能力领域命名（如Python、数据库、框架、AI等）
2. 每个域包含3-6个具体能力点
3. 能力点要具体可考察（不要泛指"编程能力"这种）
4. 区分核心域和加分域（核心的放前面）
5. 如果JD要求Redis，域名为"Redis"，能力点如["缓存策略","数据结构","集群部署"]"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        tree = json.loads(raw)
        return tree
    except:
        m = re.search(r'\{[\s\S]*\}', raw) if 'raw' in dir() else None
        if m:
            try: return json.loads(m.group())
            except: pass
        return {}

def build_candidate_profile(resume: dict) -> dict:
    """从简历提取候选人的能力画像（0-100评分）
    
    返回: {"skill_name": score, ...}
    """
    skills = resume.get("skills", [])
    projects = resume.get("projects", [])
    summary = resume.get("summary", "")
    total_years = str(resume.get("total_years", "0"))
    
    projects_text = ""
    for p in projects[:3]:
        projects_text += f"- {p.get('name','')}: {p.get('description','')[:150]}, 技术栈: {', '.join(p.get('tech_stack',[])[:5])}\n"
    
    prompt = f"""你是资深技术评估专家。请根据候选人简历，评估各项技术能力的熟练度（0-100分）。

候选人信息：
- 总结：{summary[:200]}
- 技能标签：{', '.join(skills[:15])}
- 项目经历：
{projects_text}
- 工作经验：{total_years}

请输出JSON（仅JSON，不要markdown标记）：
{{
  "技术1": 评分,
  "技术2": 评分,
  ...
}}

评分标准：
- 90-100: 精通，有多项目实战经验
- 70-89: 熟练，有项目实践
- 50-69: 了解，有学习或基础使用
- 30-49: 接触过，简历提到但无实战
- 0-29: 简历未体现

技术名用小写（如python, redis, fastapi, docker, mysql, git等）。
覆盖简历中提到的所有技术，至少输出8项。"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        profile = json.loads(raw)
        return profile
    except:
        m = re.search(r'\{[\s\S]*\}', raw) if 'raw' in dir() else None
        if m:
            try: return json.loads(m.group())
            except: pass
        return {}

def gap_analysis(capability_tree: dict, candidate_profile: dict, resume: dict, jd: dict) -> dict:
    """分析岗位要求与候选人能力的差距
    
    返回: {
        "strengths": [(domain, reason), ...],
        "weaknesses": [(domain, reason), ...],
        "risks": [(risk_desc, reason), ...],
        "focus_areas": [domain1, domain2, ...],
        "match_score": 0-100
    }
    """
    # 标准化
    def normalize_key(k): return k.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
    
    profile_norm = {normalize_key(k): v for k, v in candidate_profile.items()}
    tree_norm = {}
    for domain, abilities in capability_tree.items():
        tree_norm[normalize_key(domain)] = abilities
    
    strengths = []
    weaknesses = []
    risks = []
    focus_areas = []
    
    # 对每个JD要求的域进行评估
    for domain, abilities in tree_norm.items():
        domain_score = profile_norm.get(domain, 0)
        
        if domain_score >= 70:
            strengths.append((domain, f"得分{domain_score}，该领域较强"))
            focus_areas.append(domain)
        elif domain_score >= 40:
            weaknesses.append((domain, f"得分{domain_score}，需要加强"))
            focus_areas.append(domain)  # 需要考察但不要回避
        else:
            risks.append((domain, f"得分{domain_score}，候选人缺乏该领域经验"))
            # 缺失严重的技能不加入focus_areas（面试时不重点考察）
    
    # 从简历技能中找JD没要求但候选人有的
    extra_skills = []
    for sk, score in profile_norm.items():
        if score >= 60 and sk not in tree_norm:
            extra_skills.append(sk)
    
    if extra_skills:
        strengths.append(("额外优势", f"具备JD未要求的技能: {', '.join(extra_skills[:5])}"))
    
    # 计算匹配得分
    if tree_norm:
        total_domains = len(tree_norm)
        score_sum = sum(min(profile_norm.get(d, 0), 100) for d in tree_norm)
        match_score = round(score_sum / total_domains)
    else:
        match_score = 50
    
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risks": risks,
        "focus_areas": focus_areas,
        "match_score": match_score,
        "extra_skills": extra_skills
    }

# ===== V3 面试开始（模式驱动 + 能力差距驱动） =====

@app.post("/interview/start")
def interview_start(data: InterviewStartRequest):
    """V3 面试开始：模式驱动 + 能力差距驱动出题"""
    if data.resume_id not in resume_registry:
        return {"error": "简历不存在，请先解析简历"}
    if data.jd_id not in jd_registry:
        return {"error": "JD不存在"}

    resume = resume_registry[data.resume_id]
    jd = jd_registry[data.jd_id]
    mode = data.interview_mode or "standard"
    
    # ===== 辅助分析 =====
    level_info = detect_candidate_level(resume)
    print(f"[V3面试] 模式={mode} | 候选人={level_info['label']}")
    
    # 构建能力树和画像
    jd_content = jd.get("content", "")
    capability_tree = build_capability_tree(jd_content) if jd_content else {}
    candidate_profile = build_candidate_profile(resume)
    gap = gap_analysis(capability_tree, candidate_profile, resume, jd)
    print(f"[V3面试] 能力树域: {len(capability_tree)} | 画像技能: {len(candidate_profile)} | 匹配度: {gap['match_score']}%")
    
    session_id = f"iv_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()

    # 构建上下文
    resume_skills_str = ", ".join(resume.get('skills', [])[:15])
    jd_skills_str = ", ".join(jd.get('core_skills', []))
    projects_text = ""
    for p in resume.get('projects', [])[:3]:
        projects_text += f"- {p.get('name','')}: {p.get('description','')[:100]}, 技术栈: {', '.join(p.get('tech_stack',[])[:5])}\n"
    
    edu = resume.get('education', {})
    edu_text = f"{edu.get('degree','')} {edu.get('school','')} {edu.get('major','')}" if isinstance(edu, dict) else str(edu)

    # 模式配置
    mode_config = {
        "intern": {
            "name": "实习冲刺模式", "max_rounds": 6, "max_followups": 3,
            "persona": "你是实习面试官。友好温和，重点验证简历真实性和基础能力。不要问架构和源码问题。",
            "rules": "- 70%项目追问题 + 30%基础验证题\n- 禁止：分布式/微服务/源码/高并发\n- 重点：你做了什么、为什么这样做、遇到什么问题",
            "pressure_level": "low"
        },
        "standard": {
            "name": "校招标准模式", "max_rounds": 8, "max_followups": 4,
            "persona": "你是校招面试官。平衡基础验证和项目深挖，适当考察技术理解。",
            "rules": "- 50%项目追问题 + 30%JD能力题 + 20%基础验证题\n- 可以问概念层面（如Redis用途），不深挖源码\n- 可以问技术选型理由",
            "pressure_level": "medium"
        },
        "bigtech": {
            "name": "大厂挑战模式", "max_rounds": 10, "max_followups": 5,
            "persona": "你是大厂技术面试官。连续追问，技术深挖，考察深度和抗压能力。",
            "rules": "- 40%项目深挖 + 40%JD能力题 + 20%开放系统设计\n- 对每个技术点追3-5层\n- 要求提供具体数据、架构描述、代码示例",
            "pressure_level": "high"
        },
        "pressure": {
            "name": "压力面模式", "max_rounds": 8, "max_followups": 6,
            "persona": "你是压力面试官。高压提问，连环追问，质疑式面试。考察真实水平和抗压能力。",
            "rules": "- 50%质疑式追问 + 30%边界问题 + 20%项目验证\n- 对每个回答提出质疑\n- 连续追问直到候选人明确承认不会或给出满意答案",
            "pressure_level": "extreme"
        }
    }
    cfg = mode_config.get(mode, mode_config["standard"])

    # 构建能力差距文本
    gap_text = ""
    if gap["strengths"]:
        gap_text += "## 匹配点\n" + "\n".join(f"- ✅ {item[0]}: {item[1]}" for item in gap["strengths"][:5]) + "\n"
    if gap["weaknesses"]:
        gap_text += "## 待加强\n" + "\n".join(f"- ⚠️ {item[0]}: {item[1]}" for item in gap["weaknesses"][:5]) + "\n"
    if gap["risks"]:
        gap_text += "## 风险项\n" + "\n".join(f"- ❌ {item[0]}: {item[1]}" for item in gap["risks"][:3]) + "\n"
    
    focus_text = ", ".join(gap["focus_areas"][:5]) if gap["focus_areas"] else "项目经验"

    # 生成第一题
    first_prompt = f"""{cfg['persona']}

模式：{cfg['name']}
压力等级：{cfg['pressure_level']}
最大追问次数：{cfg['max_followups']}
出题规则：
{cfg['rules']}

## 能力差距分析
匹配度：{gap['match_score']}%
重点考察域：{focus_text}
{gap_text}

## 候选人
- 等级：{level_info['label']}
- 技能：{resume_skills_str}
- 学历：{edu_text}
- 项目：
{projects_text}

## 岗位
- {jd.get('title','')} @ {jd.get('company','')}
- 核心技能：{jd_skills_str}
- 经验要求：{jd.get('experience','')}

## 能力树
{json.dumps(capability_tree, ensure_ascii=False, indent=2)[:800]}

## 任务
生成第一道面试问题。
- 从候选人**项目经历**中切入
- 优先考察能力差距中标注的**重点考察域**
- 问题开放式、具体，引导候选人展开
- 压力面模式可以带一点质疑语气

输出JSON：
{{"question": "...", "topic": "考察的技术域", "difficulty": "基础题|项目题|加分题|开放题"}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": first_prompt}]
    )
    raw = response.choices[0].message.content.strip()
    first_question = "请介绍一下你最近做的一个项目，你负责了哪些部分？"
    first_topic = "项目经验"
    first_difficulty = "项目题"
    try:
        parsed = json.loads(raw)
        first_question = parsed.get("question", first_question)
        first_topic = parsed.get("topic", first_topic)
        first_difficulty = parsed.get("difficulty", "项目题")
    except:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                parsed = json.loads(m.group())
                first_question = parsed.get("question", first_question)
                first_topic = parsed.get("topic", first_topic)
                first_difficulty = parsed.get("difficulty", "项目题")
            except: pass

    interview_sessions[session_id] = {
        "id": session_id,
        "resume_id": data.resume_id,
        "jd_id": data.jd_id,
        "mode": mode,
        "mode_name": cfg["name"],
        "pressure_level": cfg["pressure_level"],
        "resume_name": resume.get("filename", ""),
        "jd_title": jd.get("title", ""),
        "jd_company": jd.get("company", ""),
        "current_round": 1,
        "current_topic": first_topic,
        "candidate_level": level_info["level"],
        "candidate_label": level_info["label"],
        "capability_tree": capability_tree,
        "candidate_profile": candidate_profile,
        "gap_analysis": gap,
        "follow_up_chain": [],
        "fu_count_on_topic": 0,
        "max_follow_ups_per_topic": cfg["max_followups"],
        "max_rounds": cfg["max_rounds"],
        "rounds": [{
            "round": 1, "type": "main", "topic": first_topic,
            "difficulty": first_difficulty,
            "question": first_question, "answer": None,
            "evaluation": None, "scores": None,
            "quality": None  # {completeness, accuracy, depth, authenticity}
        }],
        "accumulated_scores": {d: [] for d in SCORE_DIMS},
        "accumulated_quality": [],  # 追问质量评分
        "status": "active",
        "created_at": now
    }
    save_sessions()

    return {
        "session_id": session_id, 
        "question": first_question, 
        "topic": first_topic,
        "difficulty": first_difficulty,
        "mode": cfg["name"],
        "match_score": gap["match_score"],
        "capability_tree": {k: v[:3] for k, v in list(capability_tree.items())[:6]},
        "candidate_profile": dict(list(candidate_profile.items())[:8]),
        "gap": {"strengths": [s[0] for s in gap["strengths"][:3]], 
                "weaknesses": [w[0] for w in gap["weaknesses"][:3]]},
        "round": 1, 
        "total_rounds": cfg["max_rounds"],
        "type": "main"
    }

# ===== V3 动态追问引擎（4维质量评估） =====

# ===== V2 回答处理：动态追问引擎 =====

@app.post("/interview/answer")
def interview_answer(data: InterviewAnswerRequest):
    """V3 动态追问：4维质量评估 → 模式驱动追问"""
    if data.session_id not in interview_sessions:
        return {"error": "面试会话不存在"}
    session = interview_sessions[data.session_id]
    if session["status"] != "active":
        return {"error": "面试已结束"}

    current_round = session["current_round"]
    last_rd = session["rounds"][-1]
    last_rd["answer"] = data.answer

    # ===== 4维质量分析 =====
    word_count = len(data.answer)
    tech_keywords = []
    for kw in TECH_DRILL_CHAIN.keys():
        if kw.lower() in data.answer.lower():
            tech_keywords.append(kw)
    has_code = any(tag in data.answer.lower() for tag in ["@override", "class ", "def ", "func ", "select ", "import ", "public ", "private ", "@api"])
    has_numbers = any(c.isdigit() for c in data.answer[:200])
    has_specific = has_code or has_numbers or (word_count > 150 and tech_keywords)
    
    quality = {
        "completeness": min(100, max(10, word_count)) if word_count > 0 else 0,
        "accuracy": 80 if tech_keywords else 40,
        "depth": 90 if (has_code and word_count > 150) else (60 if tech_keywords else 20 if word_count < 50 else 40),
        "authenticity": 85 if has_specific else 30
    }
    last_rd["quality"] = quality
    session.setdefault("accumulated_quality", []).append(quality)

    resume = resume_registry.get(session["resume_id"], {})
    jd = jd_registry.get(session["jd_id"], {})

    # 对话历史
    history_text = ""
    for rd in session["rounds"]:
        if rd.get("answer"):
            t = rd.get("type","main")
            tag = "🔹追问" if t == "follow_up" else f"Q{rd['round']}"
            q = rd.get("quality", {})
            q_str = f" [完整:{q.get('completeness',0)} 准确:{q.get('accuracy',0)} 深度:{q.get('depth',0)} 真实:{q.get('authenticity',0)}]" if q else ""
            history_text += f"\n{tag}: {rd['question']}\nA: {rd['answer'][:300]}{q_str}\n"

    current_topic = session.get("current_topic", "")
    follow_up_chain = session.get("follow_up_chain", [])
    fu_count_on_topic = session.get("fu_count_on_topic", 0)
    mode = session.get("mode", "standard")
    pressure = session.get("pressure_level", "medium")

    # 模式对应的追问策略
    mode_guide = {
        "intern": "你是实习面试官。语气友好鼓励式。追问规则：回答<40字或过于笼统时追问，追问以引导为主（如'能再具体说说吗'），不质疑不施压。",
        "standard": "你是校招面试官。正常追问：回答<60字或无细节时追问。追问要求具体化。",
        "bigtech": "你是大厂面试官。深度追问：每个回答至少追问2层。要求架构描述、性能数据、代码示例。回答不充分时严厉追问。",
        "pressure": "你是压力面试官。高强度追问：每句话都要质疑。追问以'这只是表面''实际项目中不是这样的''你能证明吗'开头。连续追问直到候选人承认不会或给出有深度的回答。",
    }

    gap = session.get("gap_analysis", {})
    focus_areas = gap.get("focus_areas", [])
    
    prompt = f"""{mode_guide.get(mode, mode_guide['standard'])}

模式：{session.get('mode_name','')} | 压力等级：{pressure}

## 能力差距
- 候选人匹配度：{gap.get('match_score',0)}%
- 重点考察域：{', '.join(focus_areas[:5]) if focus_areas else '项目经验'}
- 优势：{', '.join(s[0] for s in gap.get('strengths',[])[:3])}
- 风险：{', '.join(r[0] for r in gap.get('risks',[])[:3])}

## 回答质量评估
- 字数：{word_count}
- 完整度：{quality['completeness']} | 准确度：{quality['accuracy']} | 深度：{quality['depth']} | 真实性：{quality['authenticity']}
- 技术关键词：{', '.join(tech_keywords) if tech_keywords else '无'}
- 含具体细节：{'是' if has_specific else '否'}

## 对话历史
{history_text}

## 当前
问题({last_rd.get('type','main')}): {last_rd.get('question','')}
回答: {data.answer[:1500]}

话题：{current_topic} | 该话题追问：{fu_count_on_topic}/{session['max_follow_ups_per_topic']}次

## 决策

输出JSON：
{{"evaluation": "15字点评",
 "quality": {{"completeness":0-100,"accuracy":0-100,"depth":0-100,"authenticity":0-100}},
 "scores": {{"expression":0-100,"logic":0-100,"project_authenticity":0-100,"job_match":0-100,"technical":0-100,"stress_resistance":0-100}},
 "action": "followup|next|end",
 "new_topic": "下一题话题(action=next时)",
 "question": "追问或下一题",
 "difficulty": "基础题|项目题|加分题|开放题",
 "reason": "决策理由(10字)"
}}

规则：
1. 回答<50字且无细节 → action="followup"
2. 回答有技术点且追问<{session['max_follow_ups_per_topic']}次 → action="followup"，沿技术点深挖
3. {mode}模式追问到上限后 → action="next"
4. 回答充分(质量各维>60)且追问充分 → action="next"
5. 已问{current_round}题且信息足够 → action="end"
6. **严禁随机换题**"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    try: result = json.loads(raw)
    except:
        m = re.search(r'\{[\s\S]*\}', raw)
        result = json.loads(m.group()) if m else {}

    # 更新评分
    scores = result.get("scores", {})
    last_rd["evaluation"] = result.get("evaluation", "")
    last_rd["scores"] = scores
    for dim in SCORE_DIMS:
        if dim in scores:
            session["accumulated_scores"][dim].append(scores[dim])

    action = result.get("action", "next")
    reason = result.get("reason", "")
    
    if action not in ("followup", "next", "end"):
        action = "followup" if (word_count < 50 and fu_count_on_topic < 3) else "next"

    if fu_count_on_topic >= session["max_follow_ups_per_topic"] and action == "followup":
        action = "next"
    if current_round >= session["max_rounds"]:
        action = "end"

    if action == "end":
        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()
        session["final_scores"] = {d: round(sum(v)/len(v),1) if v else 0 for d,v in session["accumulated_scores"].items()}
        session["final_scores"]["overall"] = round(sum(session["final_scores"].values())/6, 1)
        save_sessions()
        return {
            "ended": True, "type": "end",
            "evaluation": result.get("evaluation",""),
            "scores": scores, "quality": quality,
            "accumulated": session["final_scores"],
            "message": "面试结束，正在生成报告..."
        }

    if action == "followup":
        session["fu_count_on_topic"] = fu_count_on_topic + 1
        fu_q = result.get("question", "请详细说明。")
        if tech_keywords: session["follow_up_chain"] = (follow_up_chain + [tech_keywords[0]])[-5:]

        new_entry = {
            "round": current_round, "type": "followup",
            "topic": current_topic, "difficulty": result.get("difficulty", "项目题"),
            "question": fu_q, "answer": None, "evaluation": None, "scores": None, "quality": None
        }
        session["rounds"].append(new_entry)
        save_sessions()
        accum = {d: round(sum(v)/len(v),1) if v else 0 for d,v in session["accumulated_scores"].items()}
        return {
            "ended": False, "type": "followup",
            "evaluation": result.get("evaluation",""), "quality": quality,
            "scores": scores, "accumulated": accum,
            "question": fu_q, "reason": reason,
            "difficulty": result.get("difficulty", "项目题"),
            "round": current_round, "fu_count": session["fu_count_on_topic"],
            "max_fu": session["max_follow_ups_per_topic"],
            "topic": current_topic
        }

    # action == "next"
    session["current_round"] = current_round + 1
    session["fu_count_on_topic"] = 0
    session["follow_up_chain"] = []
    new_topic = result.get("new_topic", "")
    session["current_topic"] = new_topic
    next_q = result.get("question", "请继续。")
    next_diff = result.get("difficulty", "项目题")
    new_entry = {
        "round": session["current_round"], "type": "main",
        "topic": new_topic, "difficulty": next_diff,
        "question": next_q, "answer": None, "evaluation": None, "scores": None, "quality": None
    }
    session["rounds"].append(new_entry)
    save_sessions()
    accum = {d: round(sum(v)/len(v),1) if v else 0 for d,v in session["accumulated_scores"].items()}
    return {
        "ended": False, "type": "next",
        "evaluation": result.get("evaluation",""), "quality": quality,
        "scores": scores, "accumulated": accum,
        "question": next_q, "reason": reason, "difficulty": next_diff,
        "round": session["current_round"], "topic": new_topic
    }

# ===== V2 面试结束：增强报告 =====

@app.post("/interview/end")
def interview_end(data: InterviewAnswerRequest):
    """V2 结束面试，生成深度分析报告"""
    if data.session_id not in interview_sessions:
        return {"error": "会话不存在"}
    session = interview_sessions[data.session_id]
    if data.answer:
        last_rd = session["rounds"][-1]
        last_rd["answer"] = data.answer

    session["status"] = "completed"
    session["completed_at"] = datetime.now().isoformat()

    resume = resume_registry.get(session["resume_id"], {})
    jd = jd_registry.get(session["jd_id"], {})

    # 构建详细面试记录
    history_text = ""
    for rd in session["rounds"]:
        q = rd.get("question",""); a = rd.get("answer",""); e = rd.get("evaluation","")
        t = rd.get("type","main")
        topic = rd.get("topic","")
        tag = "🔹追问" if t == "follow_up" else f"Q{rd['round']}"
        topic_str = f" [{topic}]" if topic else ""
        if a:
            analysis = rd.get("answer_analysis", {})
            wc = analysis.get("word_count", 0) if analysis else len(a)
            dl = analysis.get("detail_level", "") if analysis else ""
            kw = ", ".join(analysis.get("keywords", [])[:3]) if analysis else ""
            detail_info = f" (字数:{wc}, 细节:{dl}" + (f", 关键词:{kw})" if kw else ")") if analysis else ""
            history_text += f"\n**{tag}{topic_str}**: {q}\n> 回答{detail_info}: {a[:200]}\n> 评价: {e}\n"

    final_scores = {d: round(sum(v)/len(v),1) if v else 0 for d,v in session["accumulated_scores"].items()}
    final_scores["overall"] = round(sum(final_scores.values())/6, 1)
    session["final_scores"] = final_scores

    # 收集追问统计
    followup_count = sum(1 for rd in session["rounds"] if rd.get("type") == "followup")
    main_count = sum(1 for rd in session["rounds"] if rd.get("type") == "main")
    
    # 收集深挖链
    chain_text = ""
    chain = session.get("follow_up_chain", [])
    if chain:
        chain_text = "深挖链: " + " → ".join(chain)

    report_prompt = f"""你是资深面试评估专家。基于以下面试记录生成综合报告。

## 面试概况
- 候选人：{resume.get('summary','')[:200]}
- 技能：{", ".join(resume.get('skills',[])[:10])}
- 岗位：{jd.get('title','')} @ {jd.get('company','')}
- 面试类型：{session.get('interview_type','')}
- 主问题数：{main_count}，追问数：{followup_count}
{chain_text}

## 面试记录
{history_text[:4000]}

## 评分
| 维度 | 分数 |
|------|------|
| 表达能力 | {final_scores.get('expression',0)} |
| 逻辑能力 | {final_scores.get('logic',0)} |
| 项目真实性 | {final_scores.get('project_authenticity',0)} |
| 岗位匹配度 | {final_scores.get('job_match',0)} |
| 技术能力 | {final_scores.get('technical',0)} |
| 抗压能力 | {final_scores.get('stress_resistance',0)} |
| **综合** | **{final_scores.get('overall',0)}** |

请输出Markdown格式报告：

### 一、综合评分: **{final_scores.get('overall',0)}分** / 100
(50字综合评价)

### 二、核心优势 (3-5点)
每点先标记维度再陈述。格式：**[维度]** 具体描述

### 三、核心风险 (3-5点)
具体指出回答中暴露的问题

### 四、技术深度评估
- 追问表现：在被追问时的应对能力
- 知识盲区：具体指出哪些技术点回答不充分
- 深挖链表现：在技术深挖过程中的表现

### 五、提升建议 (按优先级Top5)
具体可执行的提升计划

### 六、岗位匹配度分析
基于JD要求和面试表现的综合匹配度判断

### 七、录用建议
**录用** / **待定** / **不录用** + 理由"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": report_prompt}]
    )
    session["report"] = response.choices[0].message.content
    save_sessions()

    return {
        "ended": True,
        "final_scores": final_scores,
        "report": session["report"],
        "stats": {
            "main_questions": main_count,
            "follow_ups": followup_count,
            "chain_depth": len(chain)
        }
    }


@app.get("/interview/history")
def interview_history():
    """获取面试历史"""
    items = []
    for sid, s in interview_sessions.items():
        item = {
            "id": sid,
            "type": s.get("mode", s.get("interview_type","")),
            "mode": s.get("mode_name", ""),
            "resume_name": s.get("resume_name",""),
            "jd_title": s.get("jd_title",""),
            "jd_company": s.get("jd_company",""),
            "status": s.get("status",""),
            "score": s.get("final_scores",{}).get("overall"),
            "match_score": s.get("gap_analysis",{}).get("match_score", 0),
            "created_at": s.get("created_at",""),
            "rounds_count": len([r for r in s.get("rounds",[]) if r.get("answer")]),
            "has_report": bool(s.get("report"))
        }
        items.append(item)
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@app.get("/interview/report/{session_id}")
def interview_report_detail(session_id: str):
    """获取面试详情（含完整对话记录和元数据）"""
    if session_id not in interview_sessions:
        return {"error": "会话不存在"}
    s = interview_sessions[session_id]
    return {
        "id": s["id"],
        "report": s.get("report",""),
        "final_scores": s.get("final_scores",{}),
        "rounds": s.get("rounds",[]),
        "resume_name": s.get("resume_name",""),
        "jd_title": s.get("jd_title",""),
        "jd_company": s.get("jd_company",""),
        "mode": s.get("mode_name",""),
        "match_score": s.get("gap_analysis",{}).get("match_score", 0),
        "candidate_level": s.get("candidate_label",""),
        "candidate_profile": s.get("candidate_profile", {}),
        "capability_tree": s.get("capability_tree", {}),
        "gap_analysis": s.get("gap_analysis", {}),
        "created_at": s.get("created_at",""),
        "completed_at": s.get("completed_at",""),
        "status": s.get("status","")
    }

# ===== 生成最佳答案 =====

class BestAnswerRequest(BaseModel):
    session_id: str
    round_index: int  # rounds数组中的索引

@app.post("/interview/best_answer")
def interview_best_answer(data: BestAnswerRequest):
    """为指定问题生成优秀候选人角度的最佳答案"""
    if data.session_id not in interview_sessions:
        return {"error": "会话不存在"}
    s = interview_sessions[data.session_id]
    rounds = s.get("rounds", [])
    if data.round_index < 0 or data.round_index >= len(rounds):
        return {"error": "题目索引超出范围"}
    
    rd = rounds[data.round_index]
    question = rd.get("question", "")
    user_answer = rd.get("answer", "")
    evaluation = rd.get("evaluation", "")
    scores = rd.get("scores", {})
    quality = rd.get("quality", {})
    
    jd = jd_registry.get(s.get("jd_id", ""), {})
    resume = resume_registry.get(s.get("resume_id", ""), {})
    
    prompt = f"""你是资深技术面试官。一位候选人在面试中回答了一道题，表现不够理想。
请生成一份"如果是一个优秀的候选人来回答"的参考答案。

## 题目
{question}

## 候选人的回答（不够理想）
{user_answer[:800]}

## 面试官点评
{evaluation}

## 岗位背景
- 职位：{jd.get('title','')} @ {jd.get('company','')}
- 核心技能：{', '.join(jd.get('core_skills',[])[:8])}

## 要求
1. 结构清晰，分点阐述
2. 包含具体的技术细节或操作步骤
3. 如果有数据/性能指标，给出合理数值
4. 控制在300字以内
5. 语气自然，像是口述面试回答

输出格式（纯文本，不要JSON）：
直接输出改进后的参考答案。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "question": question,
        "user_answer": user_answer,
        "best_answer": response.choices[0].message.content.strip()
    }

# ====== 图片导入（OCR） ======

import base64
from PIL import Image
import io
import os as _os

# OCR引擎选择：优先PaddleOCR，降级RapidOCR
_ocr_engine = None
_ocr_mode = None

def get_ocr():
    global _ocr_engine, _ocr_mode
    if _ocr_engine is not None:
        return _ocr_engine

    # 尝试1: PaddleOCR
    try:
        _os.environ.setdefault("PADDLE_HOME", os.path.join(os.getcwd(), ".paddle_cache"))
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(lang="ch")
        _ocr_mode = "paddle"
        print("[OCR] Using PaddleOCR")
        return _ocr_engine
    except Exception as e:
        print(f"[OCR] PaddleOCR unavailable: {e}")

    # 尝试2: EasyOCR
    try:
        import easyocr
        model_dir = os.path.join(os.getcwd(), ".easyocr_models")
        _os.makedirs(model_dir, exist_ok=True)
        _ocr_engine = easyocr.Reader(["ch_sim", "en"], gpu=False, model_storage_directory=model_dir)
        _ocr_mode = "easyocr"
        print("[OCR] Using EasyOCR")
        return _ocr_engine
    except Exception as e:
        print(f"[OCR] EasyOCR unavailable: {e}")

    # 降级: RapidOCR (最兼容)
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        _ocr_mode = "rapidocr"
        print("[OCR] Using RapidOCR")
        return _ocr_engine
    except Exception as e:
        print(f"[OCR] RapidOCR unavailable: {e}")
        raise RuntimeError("No OCR engine available. Install paddleocr, easyocr, or rapidocr-onnxruntime")


def run_ocr(image_path: str) -> str:
    """执行OCR，返回识别文本"""
    ocr = get_ocr()

    if _ocr_mode == "paddle":
        try:
            r = ocr.ocr(image_path)
            if r and r[0]:
                return "\n".join([item[1][0] for item in r[0] if item[1][0].strip()])
        except:
            try:
                r = ocr.ocr(image_path, cls=False)
                if r and r[0]:
                    return "\n".join([item[1][0] for item in r[0] if item[1][0].strip()])
            except:
                pass
        return ""

    elif _ocr_mode == "easyocr":
        r = ocr.readtext(image_path)
        return "\n".join([item[1] for item in r if item[1].strip()])

    else:  # rapidocr
        result, _ = ocr(image_path)
        if result:
            return "\n".join([item[1] for item in result if item[1] and item[1].strip()])
        return ""


class ImportConfirmRequest(BaseModel):
    ocr_text: str
    import_type: str  # "jd" or "resume"
    title: str = ""
    company: str = ""


@app.post("/import/ocr")
async def import_ocr(files: list[UploadFile] = File(...)):
    """上传图片，OCR识别返回预览文本"""
    if not files:
        return {"error": "请上传至少一张图片"}

    all_texts = []
    file_names = []
    for file in files:
        fname = (file.filename or "").lower()
        ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
        if ext not in ("png", "jpg", "jpeg", "webp", "bmp"):
            return {"error": f"不支持的文件类型: {ext}，支持 PNG/JPG/JPEG/WEBP/BMP"}

        img_bytes = await file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            tmp_path = f"temp_ocr_{uuid.uuid4().hex[:6]}.jpg"
            img.save(tmp_path, "JPEG", quality=90)
        except Exception as e:
            return {"error": f"图片读取失败: {str(e)}"}

        try:
            text = run_ocr(tmp_path)
            all_texts.append(text)
        except Exception as e:
            all_texts.append(f"[OCR失败: {str(e)}]")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        file_names.append(file.filename or "unknown")

    merged = "\n---\n".join([t for t in all_texts if t.strip()])

    # 自动判断JD/简历类型
    detected_type = "jd"
    jd_kw = ["岗位", "职责", "要求", "任职", "招聘", "薪资", "福利", "学历", "经验", "职位", "J", "D"]
    resume_kw = ["简历", "教育", "学校", "专业", "实习", "项目经验", "自我评价", "求职", "意向", "电话", "邮箱"]
    jd_score = sum(1 for kw in jd_kw if kw in merged)
    resume_score = sum(1 for kw in resume_kw if kw in merged)
    if resume_score > jd_score:
        detected_type = "resume"

    return {
        "text": merged,
        "page_count": len(files),
        "page_texts": all_texts,
        "file_names": file_names,
        "char_count": len(merged),
        "detected_type": detected_type,
        "ocr_engine": _ocr_mode
    }


@app.post("/import/confirm")
def import_confirm(data: ImportConfirmRequest):
    """确认OCR结果，导入JD或简历"""
    text = data.ocr_text.strip()
    if not text:
        return {"error": "文本为空"}

    if data.import_type == "jd":
        jd_id = f"jd_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()
        jd_registry[jd_id] = {
            "id": jd_id, "title": data.title or "未命名JD",
            "company": data.company or "未填写",
            "content": text, "core_skills": [], "bonus_skills": [],
            "benefits": [], "education": "", "experience": "",
            "graduation_year": None, "interview_focus": [],
            "filename": "图片导入", "source": "image",
            "quality_score": 0, "created_at": now, "updated_at": now
        }
        save_jd_registry()
        try:
            parsed = parse_jd_with_ai(text)
            jd_registry[jd_id].update({
                "title": parsed.get("title") or jd_registry[jd_id]["title"],
                "company": parsed.get("company") or jd_registry[jd_id]["company"],
                "core_skills": parsed.get("core_skills", []),
                "bonus_skills": parsed.get("bonus_skills", []),
                "benefits": parsed.get("benefits", []),
                "education": parsed.get("education", ""),
                "experience": parsed.get("experience", ""),
                "graduation_year": parsed.get("graduation_year"),
                "interview_focus": parsed.get("interview_focus", []),
                "updated_at": datetime.now().isoformat()
            })
            jd_registry[jd_id]["quality_score"] = compute_jd_quality(jd_registry[jd_id])
            save_jd_registry()
        except:
            pass
        return {"message": "JD导入成功", "jd": jd_registry[jd_id]}

    elif data.import_type == "resume":
        coll_name = f"pdf_{uuid.uuid4().hex[:8]}"
        chunk_size = 500; overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

        pdf_registry[coll_name] = {"filename": "图片导入简历", "chunks": len(chunks)}
        with open("pdf_registry.json", "w", encoding="utf-8") as f:
            json.dump(pdf_registry, f, ensure_ascii=False, indent=4)

        pdf_cache[coll_name] = chunks
        build_bm25(coll_name, chunks)

        for chunk in chunks:
            if not chunk.strip(): continue
            embedding = embedding_model.encode(chunk).tolist()
            collection.add(ids=[str(uuid.uuid4())], documents=[chunk],
                          embeddings=[embedding],
                          metadatas=[{"collection": coll_name}])

        try:
            parsed = parse_resume_with_ai(text)
            resume_id = f"res_{coll_name}"
            resume_registry[resume_id] = {
                "id": resume_id, "pdf_collection": coll_name,
                "filename": "图片导入简历", "skills": parsed.get("skills", []),
                "projects": parsed.get("projects", []),
                "education": parsed.get("education", {}),
                "internships": parsed.get("internships", []),
                "certificates": parsed.get("certificates", []),
                "total_years": parsed.get("total_years", "0"),
                "summary": parsed.get("summary", ""),
                "raw_text": text, "parsed_at": datetime.now().isoformat()
            }
            save_resume_registry()
        except:
            pass

        return {"message": "简历导入成功", "collection": coll_name, "chunks": len(chunks)}

    return {"error": f"未知类型: {data.import_type}"}
