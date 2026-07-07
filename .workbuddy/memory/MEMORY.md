# 项目记忆 — FastAPI RAG 知识库

## 技术栈
- FastAPI（单文件架构，HTML内嵌）
- DeepSeek API（deepseek-chat）
- ChromaDB（单 collection: knowledge_base，通过 metadata.collection 区分PDF）
- SentenceTransformer（all-MiniLM-L6-v2）
- pypdf（PDF解析）

## 已完成功能
- [x] PDF上传 + 自动分块 + Embedding
- [x] PDF问答（RAG QA模式）
- [x] PDF全文总结（summary模式）
- [x] 简历分析
- [x] PDF管理中心（查看/删除）
- [x] **多PDF知识库问答**（2026-06-11）：复选框多选，ChromaDB `$in` 过滤，来源追踪
- [x] **检索来源显示优化**（2026-06-11）：文件分组、距离排序、三级相关度标签、折叠展开、命中统计
- [x] **检索相似度显示**（2026-06-11）：L2距离→余弦相似度百分比，醒目徽章展示
- [x] **智能知识库搜索**（2026-06-11）：搜索框实时过滤PDF列表
- [x] **知识库统计面板**（2026-06-11）：PDF/Chunk/向量三指标，实时刷新
- [x] **JD岗位匹配分析**（2026-06-11）：简历+JD全文对比，7维度DeepSeek分析，匹配度/缺失技能/通过率
- [x] **导航结构重构**（2026-06-12）：5项导航（首页/资料中心/AI分析中心/AI面试中心/系统设置），section-page模式 + 内部Tab切换
- [x] **JD管理中心 V1**（2026-06-12）：完整CRUD + AI解析结构化JD + 卡片布局 + 详情弹窗 + 文本/PDF双模式创建。预留 match_resume_to_jd()
- [x] **简历-JD匹配引擎 V2**（2026-06-12）：结构化匹配（技能50%+项目25%+学历10%+经验15%加权），AI简历解析，技能覆盖柱状图可视化，POST /match_v2

## 架构约定
- 不要引入 React/Vue
- 不要前后端分离
- 保持单文件 main.py 结构
- 每次只实现一个功能
- ChromaDB 距离: `l2`(平方欧氏)，公式 `sim = max(0, 1-dist/2) * 100`
- **前端布局**：侧边栏 + 5个section-page（2026-06-12重构），通过 showSection() 切换一级导航，资料中心/AI分析中心内部用 Tab（switchDCTab/switchAnTab）切换
