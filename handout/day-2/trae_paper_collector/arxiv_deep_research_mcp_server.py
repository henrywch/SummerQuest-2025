import json
import os
import re
import asyncio
import aiohttp
import sqlite3
import hashlib
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from fastmcp import FastMCP
from enum import Enum
from collections import defaultdict
from PyPDF2 import PdfReader
from io import BytesIO

# 创建 FastMCP 服务器实例
mcp = FastMCP("deep-research-system")

# ======================== 数据模型 ========================
class PaperSource(Enum):
    ARXIV = "arXiv"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    ZOTERO = "Zotero"
    USER_UPLOAD = "User Upload"

@dataclass
class Author:
    name: str
    affiliation: str = ""
    author_id: str = ""

@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    url: str
    authors: List[Author]
    published: str
    source: PaperSource
    citations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    file_path: str = ""
    embeddings: List[float] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "authors": [{"name": a.name, "affiliation": a.affiliation} for a in self.authors],
            "published": self.published,
            "source": self.source.value,
            "citations": self.citations,
            "references": self.references,
            "tags": self.tags,
            "notes": self.notes,
            "file_path": self.file_path
        }

@dataclass
class ResearchProject:
    project_id: str
    name: str
    description: str
    papers: List[str] = field(default_factory=list)  # 存储 paper_id
    notes: Dict[str, str] = field(default_factory=dict)  # paper_id -> notes
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    graph_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "papers": self.papers,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

@dataclass
class ResearchSummary:
    summary_id: str
    project_id: str
    content: str
    key_findings: List[str]
    research_gaps: List[str]
    future_directions: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

# ======================== 核心系统 ========================
class DeepResearchSystem:
    def __init__(self, db_path: str = "research.db", debug: bool = False):
        self.debug = debug
        self.db_path = db_path
        self._init_db()
        self.semantic_scholar_api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.zotero_api_key = os.getenv('ZOTERO_API_KEY', '14AQV7s2juLVLRnUbERAC29P')
        self.zotero_user_id = os.getenv('ZOTERO_USER_ID', 'henrywch')
        self.paper_storage = os.path.join(os.path.dirname(__file__), "paper_storage")
        
        # 确保存储目录存在
        os.makedirs(self.paper_storage, exist_ok=True)
        
        if self.debug:
            print(f"[DEBUG] 初始化DeepResearchSystem，数据库路径: {db_path}")
            print(f"[DEBUG] 论文存储目录: {self.paper_storage}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建论文表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    url TEXT,
                    authors TEXT,
                    published TEXT,
                    source TEXT,
                    citations TEXT,
                    references TEXT,
                    tags TEXT,
                    notes TEXT,
                    file_path TEXT,
                    embeddings BLOB
                );
            ''')
            
            # 创建项目表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    papers TEXT,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    graph_data TEXT
                );
            ''')
            
            # 创建研究摘要表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_summaries (
                    summary_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    content TEXT,
                    key_findings TEXT,
                    research_gaps TEXT,
                    future_directions TEXT,
                    created_at TEXT
                );
            ''')
            
            conn.commit()

    # ================= 论文管理功能 =================
    def save_paper(self, paper: Paper):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            authors_json = json.dumps([asdict(a) for a in paper.authors])
            citations_json = json.dumps(paper.citations)
            references_json = json.dumps(paper.references)
            tags_json = json.dumps(paper.tags)
            embeddings_blob = sqlite3.Binary(json.dumps(paper.embeddings).encode('utf-8'))
            
            cursor.execute('''
                INSERT OR REPLACE INTO papers (
                    paper_id, title, abstract, url, authors, published, source,
                    citations, references, tags, notes, file_path, embeddings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper.paper_id,
                paper.title,
                paper.abstract,
                paper.url,
                authors_json,
                paper.published,
                paper.source.value,
                citations_json,
                references_json,
                tags_json,
                paper.notes,
                paper.file_path,
                embeddings_blob
            ))
            conn.commit()
            
            if self.debug:
                print(f"[DEBUG] 保存论文: {paper.title} ({paper.paper_id})")

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,))
            row = cursor.fetchone()
            
            if row:
                authors = [Author(**a) for a in json.loads(row[4])]
                citations = json.loads(row[7])
                references = json.loads(row[8])
                tags = json.loads(row[9])
                embeddings = json.loads(row[12].decode('utf-8')) if row[12] else []
                
                return Paper(
                    paper_id=row[0],
                    title=row[1],
                    abstract=row[2],
                    url=row[3],
                    authors=authors,
                    published=row[5],
                    source=PaperSource(row[6]),
                    citations=citations,
                    references=references,
                    tags=tags,
                    notes=row[10],
                    file_path=row[11],
                    embeddings=embeddings
                )
        return None

    # ================= 项目管理功能 =================
    def create_project(self, name: str, description: str) -> ResearchProject:
        project_id = f"project_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        project = ResearchProject(project_id, name, description)
        self.save_project(project)
        return project

    def save_project(self, project: ResearchProject):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            papers_json = json.dumps(project.papers)
            notes_json = json.dumps(project.notes)
            graph_data_json = json.dumps(project.graph_data) if project.graph_data else "{}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO projects (
                    project_id, name, description, papers, notes, created_at, updated_at, graph_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project.project_id,
                project.name,
                project.description,
                papers_json,
                notes_json,
                project.created_at,
                project.updated_at,
                graph_data_json
            ))
            conn.commit()
            
            if self.debug:
                print(f"[DEBUG] 保存项目: {project.name} ({project.project_id})")

    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM projects WHERE project_id = ?', (project_id,))
            row = cursor.fetchone()
            
            if row:
                papers = json.loads(row[3])
                notes = json.loads(row[4])
                graph_data = json.loads(row[7]) if row[7] else {}
                
                return ResearchProject(
                    project_id=row[0],
                    name=row[1],
                    description=row[2],
                    papers=papers,
                    notes=notes,
                    created_at=row[5],
                    updated_at=row[6],
                    graph_data=graph_data
                )
        return None

    def add_paper_to_project(self, project_id: str, paper_id: str):
        project = self.get_project(project_id)
        if project and paper_id not in project.papers:
            project.papers.append(paper_id)
            project.updated_at = datetime.now().isoformat()
            self.save_project(project)
            
            if self.debug:
                print(f"[DEBUG] 添加论文 {paper_id} 到项目 {project_id}")

    # ================= 研究分析功能 =================
    async def analyze_project(self, project_id: str) -> ResearchSummary:
        """使用AI分析整个研究项目"""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"项目 {project_id} 不存在")
        
        # 获取项目中的所有论文
        papers = [self.get_paper(pid) for pid in project.papers]
        papers = [p for p in papers if p is not None]
        
        if not papers:
            raise ValueError("项目中没有论文可供分析")
        
        if self.debug:
            print(f"[DEBUG] 开始分析项目: {project.name} ({len(papers)}篇论文)")
        
        # 生成提示词
        prompt = self._generate_analysis_prompt(papers, project)
        
        # 调用AI分析
        analysis_result = await self._call_ai_analysis(prompt)
        
        # 解析结果
        summary_id = f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        summary = ResearchSummary(
            summary_id=summary_id,
            project_id=project_id,
            content=analysis_result.get("summary", ""),
            key_findings=analysis_result.get("key_findings", []),
            research_gaps=analysis_result.get("research_gaps", []),
            future_directions=analysis_result.get("future_directions", [])
        )
        
        # 保存摘要
        self._save_research_summary(summary)
        
        # 更新项目图谱数据
        self._update_project_graph(project_id, analysis_result.get("graph_data", {}))
        
        return summary

    def _generate_analysis_prompt(self, papers: List[Paper], project: ResearchProject) -> str:
        """生成AI分析提示词"""
        prompt = f"""你是一个专业的研究助理，正在分析一个研究项目。请基于以下论文提供深入分析：

项目名称: {project.name}
项目描述: {project.description}

论文列表:
"""
        for i, paper in enumerate(papers, 1):
            prompt += f"\n{i}. {paper.title} ({paper.published})\n"
            prompt += f"   作者: {', '.join([a.name for a in paper.authors])}\n"
            prompt += f"   摘要: {paper.abstract[:300]}...\n"
        
        prompt += """
请提供以下内容：
1. 综合摘要：总结整个研究项目的核心主题和主要发现（500字以内）
2. 关键发现：列出3-5个最重要的研究发现
3. 研究空白：识别2-3个研究空白或未解决的问题
4. 未来方向：提出3-5个未来研究方向
5. 图谱数据：返回一个JSON格式的引文图谱数据，包含节点和边

请使用以下JSON格式返回结果：
{
    "summary": "综合摘要内容",
    "key_findings": ["发现1", "发现2", ...],
    "research_gaps": ["空白1", "空白2", ...],
    "future_directions": ["方向1", "方向2", ...],
    "graph_data": {
        "nodes": [
            {"id": "paper_id1", "label": "论文标题1", "group": "主题类别"},
            ...
        ],
        "edges": [
            {"from": "paper_id1", "to": "paper_id2", "label": "引用关系"},
            ...
        ]
    }
}
"""
        return prompt

    async def _call_ai_analysis(self, prompt: str) -> Dict[str, Any]:
        """调用AI进行分析（这里模拟实现，实际应调用OpenAI API或本地LLM）"""
        if self.debug:
            print(f"[DEBUG] 调用AI分析，提示长度: {len(prompt)}字符")
        
        # 模拟AI响应
        await asyncio.sleep(2)  # 模拟网络延迟
        
        # 在实际应用中，这里应该调用真实的AI API
        # 示例：使用OpenAI GPT-4
        # async with aiohttp.ClientSession() as session:
        #     headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        #     payload = {
        #         "model": "gpt-4",
        #         "messages": [{"role": "user", "content": prompt}],
        #         "temperature": 0.7
        #     }
        #     async with session.post("https://api.openai.com/v1/chat/completions", 
        #                            json=payload, headers=headers) as response:
        #         data = await response.json()
        #         content = data['choices'][0]['message']['content']
        #         return json.loads(content)
        
        # 模拟响应
        return {
            "summary": "该项目研究了深度学习在自然语言处理中的应用。主要关注点包括Transformer架构的优化、预训练语言模型的效率提升以及少样本学习技术。",
            "key_findings": [
                "Transformer架构在多个NLP任务中表现出色",
                "模型压缩技术可以将大型模型尺寸减少60%而性能损失小于5%",
                "提示工程显著提高少样本学习性能"
            ],
            "research_gaps": [
                "缺乏对多语言模型的统一评估框架",
                "模型解释性研究不足",
                "实际部署中的效率问题研究较少"
            ],
            "future_directions": [
                "开发更高效的多语言预训练框架",
                "研究模型压缩与解释性的平衡",
                "探索领域自适应在实际应用中的表现"
            ],
            "graph_data": {
                "nodes": [
                    {"id": "paper1", "label": "论文A", "group": "Transformer优化"},
                    {"id": "paper2", "label": "论文B", "group": "模型压缩"},
                    {"id": "paper3", "label": "论文C", "group": "少样本学习"}
                ],
                "edges": [
                    {"from": "paper1", "to": "paper2", "label": "引用"},
                    {"from": "paper2", "to": "paper3", "label": "引用"},
                    {"from": "paper3", "to": "paper1", "label": "相关"}
                ]
            }
        }

    def _save_research_summary(self, summary: ResearchSummary):
        """保存研究摘要到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            key_findings_json = json.dumps(summary.key_findings)
            research_gaps_json = json.dumps(summary.research_gaps)
            future_directions_json = json.dumps(summary.future_directions)
            
            cursor.execute('''
                INSERT INTO research_summaries (
                    summary_id, project_id, content, key_findings, research_gaps, future_directions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary.summary_id,
                summary.project_id,
                summary.content,
                key_findings_json,
                research_gaps_json,
                future_directions_json,
                summary.created_at
            ))
            conn.commit()
            
            if self.debug:
                print(f"[DEBUG] 保存研究摘要: {summary.summary_id}")

    def _update_project_graph(self, project_id: str, graph_data: Dict[str, Any]):
        """更新项目的图谱数据"""
        project = self.get_project(project_id)
        if project:
            project.graph_data = graph_data
            self.save_project(project)
            
            if self.debug:
                print(f"[DEBUG] 更新项目图谱: {project_id}")

    # ================= 文献获取功能 =================
    async def get_arxiv_paper(self, arxiv_id: str) -> Optional[Paper]:
        """从arXiv获取论文"""
        async with aiohttp.ClientSession() as session:
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self.parse_arxiv_response(content, arxiv_id)
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] 获取arXiv论文失败: {e}")
        return None

    def parse_arxiv_response(self, xml_content: str, arxiv_id: str) -> Optional[Paper]:
        """解析arXiv API响应（简化版）"""
        # 实际实现应使用XML解析
        # 这里返回模拟数据
        return Paper(
            paper_id=f"arxiv_{arxiv_id}",
            title="深度学习在自然语言处理中的最新进展",
            abstract="本文综述了深度学习在NLP领域的最新进展，包括Transformer架构、预训练模型等...",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            authors=[
                Author(name="张伟", affiliation="清华大学"),
                Author(name="李芳", affiliation="北京大学")
            ],
            published="2023-01-15",
            source=PaperSource.ARXIV
        )

    async def get_semantic_scholar_paper(self, paper_id: str) -> Optional[Paper]:
        """从Semantic Scholar获取论文"""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
            params = {"fields": "title,abstract,url,authors,publicationDate,externalIds"}
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.parse_semantic_scholar_response(data)
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] 获取Semantic Scholar论文失败: {e}")
        return None

    def parse_semantic_scholar_response(self, data: Dict) -> Optional[Paper]:
        """解析Semantic Scholar响应"""
        arxiv_id = data.get("externalIds", {}).get("ArXiv", "")
        paper_id = data.get("paperId", "")
        
        if not paper_id:
            return None
        
        return Paper(
            paper_id=f"ss_{paper_id}",
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            url=data.get("url", ""),
            authors=[Author(name=author["name"]) for author in data.get("authors", [])],
            published=data.get("publicationDate", ""),
            source=PaperSource.SEMANTIC_SCHOLAR,
            citations=[],
            references=[]
        )

    # ================= Zotero 集成 =================
    async def sync_with_zotero(self, project_id: str):
        """从Zotero同步文献到项目"""
        if not self.zotero_api_key or not self.zotero_user_id:
            raise ValueError("Zotero API密钥或用户ID未配置")
        
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"项目 {project_id} 不存在")
        
        if self.debug:
            print(f"[DEBUG] 开始从Zotero同步项目: {project.name}")
        
        # 获取Zotero库中的所有项目
        zotero_items = await self._get_zotero_collection_items()
        
        # 处理每个项目
        for item in zotero_items:
            paper = self._create_paper_from_zotero_item(item)
            if paper:
                self.save_paper(paper)
                self.add_paper_to_project(project_id, paper.paper_id)
                
                # 下载附件（如果存在）
                await self._download_zotero_attachments(item, paper)
        
        return {"status": "success", "added_papers": len(zotero_items)}

    async def _get_zotero_collection_items(self) -> List[Dict]:
        """获取Zotero库中的项目（简化版）"""
        # 实际实现应调用Zotero API
        # 返回模拟数据
        return [
            {
                "key": "ABC123",
                "data": {
                    "title": "深度学习在医疗影像分析中的应用",
                    "abstractNote": "本文探讨了深度学习技术在医疗影像分析中的最新应用...",
                    "url": "https://www.example.com/paper1",
                    "creators": [
                        {"name": "王明", "affiliation": "上海交通大学医学院"},
                        {"name": "陈华", "affiliation": "复旦大学附属医院"}
                    ],
                    "date": "2022-09-10",
                    "attachments": [
                        {"key": "ATT123", "title": "fulltext.pdf"}
                    ]
                }
            },
            {
                "key": "DEF456",
                "data": {
                    "title": "基于Transformer的医学报告生成系统",
                    "abstractNote": "本文提出了一种基于Transformer架构的医学报告自动生成系统...",
                    "url": "https://www.example.com/paper2",
                    "creators": [
                        {"name": "张强", "affiliation": "浙江大学"},
                        {"name": "刘芳", "affiliation": "中山大学"}
                    ],
                    "date": "2023-03-15",
                    "attachments": [
                        {"key": "ATT456", "title": "fulltext.pdf"}
                    ]
                }
            }
        ]

    def _create_paper_from_zotero_item(self, item: Dict) -> Optional[Paper]:
        """从Zotero项目创建Paper对象"""
        data = item.get("data", {})
        if not data.get("title"):
            return None
        
        return Paper(
            paper_id=f"zotero_{item['key']}",
            title=data.get("title", ""),
            abstract=data.get("abstractNote", ""),
            url=data.get("url", ""),
            authors=[Author(name=creator["name"], affiliation=creator.get("affiliation", "")) 
                     for creator in data.get("creators", [])],
            published=data.get("date", ""),
            source=PaperSource.ZOTERO,
            tags=data.get("tags", [])
        )

    async def _download_zotero_attachments(self, item: Dict, paper: Paper):
        """下载Zotero附件（PDF）"""
        attachments = item.get("data", {}).get("attachments", [])
        if not attachments:
            return
        
        # 下载第一个附件（假设是PDF）
        attachment = attachments[0]
        attachment_key = attachment.get("key")
        
        if not attachment_key:
            return
        
        # 实际实现应调用Zotero API下载附件
        # 这里模拟下载
        pdf_content = b"%PDF-1.4..."  # 模拟PDF内容
        
        # 保存PDF文件
        filename = f"zotero_{attachment_key}.pdf"
        filepath = os.path.join(self.paper_storage, filename)
        
        with open(filepath, "wb") as f:
            f.write(pdf_content)
        
        # 更新论文记录
        paper.file_path = filepath
        self.save_paper(paper)
        
        if self.debug:
            print(f"[DEBUG] 下载并保存附件: {filepath}")

    # ================= 用户上传功能 =================
    async def upload_paper(self, file_content: bytes, filename: str) -> Paper:
        """用户上传论文PDF"""
        # 保存文件
        file_hash = hashlib.md5(file_content).hexdigest()
        paper_id = f"upload_{file_hash}"
        filepath = os.path.join(self.paper_storage, f"{paper_id}.pdf")
        
        with open(filepath, "wb") as f:
            f.write(file_content)
        
        # 提取元数据
        title, authors, abstract = self._extract_pdf_metadata(file_content)
        
        paper = Paper(
            paper_id=paper_id,
            title=title or filename,
            abstract=abstract or "",
            url="",
            authors=[Author(name=a) for a in authors],
            published=datetime.now().strftime("%Y-%m-%d"),
            source=PaperSource.USER_UPLOAD,
            file_path=filepath
        )
        
        self.save_paper(paper)
        
        # 生成嵌入向量
        await self.generate_embeddings(paper_id)
        
        return paper

    def _extract_pdf_metadata(self, pdf_content: bytes) -> Tuple[str, List[str], str]:
        """从PDF提取元数据（简化版）"""
        try:
            reader = PdfReader(BytesIO(pdf_content))
            metadata = reader.metadata
            title = metadata.get("/Title", "") if metadata else ""
            author = metadata.get("/Author", "") if metadata else ""
            authors = [author] if author else []
            
            # 提取第一页作为摘要（简化处理）
            abstract = ""
            if len(reader.pages) > 0:
                abstract = reader.pages[0].extract_text()[:500]
            
            return title, authors, abstract
        except Exception as e:
            if self.debug:
                print(f"[ERROR] PDF元数据提取失败: {e}")
            return "", [], ""

    # ================= 嵌入向量功能 =================
    async def generate_embeddings(self, paper_id: str):
        """为论文生成嵌入向量（简化版）"""
        paper = self.get_paper(paper_id)
        if not paper:
            return
        
        # 实际实现应使用嵌入模型（如OpenAI text-embedding-ada-002）
        # 这里模拟生成
        embedding = [0.1 * (i % 10) for i in range(1536)]  # 模拟1536维向量
        
        paper.embeddings = embedding
        self.save_paper(paper)
        
        if self.debug:
            print(f"[DEBUG] 生成嵌入向量: {paper_id}")

    async def find_similar_papers(self, paper_id: str, top_k: int = 5) -> List[Paper]:
        """查找相似论文"""
        target_paper = self.get_paper(paper_id)
        if not target_paper or not target_paper.embeddings:
            return []
        
        # 获取所有论文
        all_papers = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT paper_id FROM papers')
            all_paper_ids = [row[0] for row in cursor.fetchall()]
        
        all_papers = [self.get_paper(pid) for pid in all_paper_ids]
        all_papers = [p for p in all_papers if p and p.embeddings and p.paper_id != paper_id]
        
        # 计算相似度（简化版）
        # 实际实现应使用余弦相似度等度量
        similarities = []
        for paper in all_papers:
            # 模拟相似度计算
            sim = sum(a * b for a, b in zip(target_paper.embeddings, paper.embeddings))
            similarities.append((paper, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [paper for paper, _ in similarities[:top_k]]

    # ================= 可视化功能 =================
    def generate_citation_graph(self, project_id: str, output_path: str = None):
        """生成引文关系图"""
        project = self.get_project(project_id)
        if not project:
            return None
        
        # 创建图
        G = nx.DiGraph()
        
        # 添加节点
        for paper_id in project.papers:
            paper = self.get_paper(paper_id)
            if paper:
                G.add_node(paper_id, label=paper.title[:30] + "...", group=paper.tags[0] if paper.tags else "default")
        
        # 添加边
        for paper_id in project.papers:
            paper = self.get_paper(paper_id)
            if paper:
                for ref_id in paper.references:
                    if ref_id in G.nodes:
                        G.add_edge(paper_id, ref_id)
                for cit_id in paper.citations:
                    if cit_id in G.nodes:
                        G.add_edge(cit_id, paper_id)
        
        # 绘制图形
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # 按组着色
        groups = set(nx.get_node_attributes(G, 'group').values())
        color_map = plt.cm.tab10(range(len(groups)))
        group_colors = {group: color_map[i] for i, group in enumerate(groups)}
        
        node_colors = [group_colors[G.nodes[n]['group']] for n in G.nodes]
        
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # 添加图例
        legend_handles = []
        for group, color in group_colors.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=group))
        plt.legend(handles=legend_handles)
        
        plt.title(f"引文网络: {project.name}")
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    # ================= 研究策略生成 =================
    async def generate_research_strategy(self, project_id: str) -> Dict[str, Any]:
        """生成研究策略计划"""
        project = self.get_project(project_id)
        if not project:
            return {"error": "项目不存在"}
        
        # 获取项目摘要
        summaries = self._get_project_summaries(project_id)
        if not summaries:
            # 如果没有摘要，先进行分析
            summary = await self.analyze_project(project_id)
            summaries = [summary]
        
        # 生成提示
        prompt = f"""你是一个研究策略专家，请为以下研究项目制定详细的研究策略计划：

项目名称: {project.name}
项目描述: {project.description}

研究摘要:
{summaries[0].content}

研究空白:
{', '.join(summaries[0].research_gaps)}

请提供：
1. 研究目标：3-5个具体的研究目标
2. 方法论：建议的研究方法和技术路线
3. 时间规划：分阶段的时间规划表（3-6个月）
4. 资源需求：所需的研究资源（数据、计算资源、合作等）
5. 风险评估：可能的风险及应对策略

请使用以下JSON格式返回结果：
{
    "research_goals": ["目标1", "目标2", ...],
    "methodology": "研究方法描述",
    "timeline": [
        {"phase": "阶段1", "duration": "1个月", "tasks": ["任务1", "任务2"]},
        ...
    ],
    "resource_requirements": ["资源1", "资源2", ...],
    "risk_assessment": ["风险1: 应对策略", ...]
}
"""
        # 调用AI生成策略
        strategy = await self._call_ai_strategy_generation(prompt)
        return strategy

    def _get_project_summaries(self, project_id: str) -> List[ResearchSummary]:
        """获取项目的研究摘要"""
        summaries = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM research_summaries WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
            rows = cursor.fetchall()
            
            for row in rows:
                summaries.append(ResearchSummary(
                    summary_id=row[0],
                    project_id=row[1],
                    content=row[2],
                    key_findings=json.loads(row[3]),
                    research_gaps=json.loads(row[4]),
                    future_directions=json.loads(row[5]),
                    created_at=row[6]
                ))
        return summaries

    async def _call_ai_strategy_generation(self, prompt: str) -> Dict[str, Any]:
        """调用AI生成研究策略（模拟实现）"""
        await asyncio.sleep(2)  # 模拟网络延迟
        
        # 模拟响应
        return {
            "research_goals": [
                "开发多语言统一评估框架",
                "研究模型压缩与解释性的平衡方法",
                "探索领域自适应在医疗领域的应用"
            ],
            "methodology": "本研究将采用混合研究方法：1) 文献综述分析现有技术；2) 设计并实现多语言评估框架；3) 在多个数据集上进行实验验证；4) 开发模型压缩与解释性平衡算法",
            "timeline": [
                {
                    "phase": "文献调研与方案设计",
                    "duration": "1个月",
                    "tasks": [
                        "收集相关文献",
                        "分析现有方法优缺点",
                        "制定详细技术方案"
                    ]
                },
                {
                    "phase": "系统开发与实现",
                    "duration": "2个月",
                    "tasks": [
                        "开发评估框架核心模块",
                        "实现模型压缩算法",
                        "设计解释性可视化工具"
                    ]
                },
                {
                    "phase": "实验评估与优化",
                    "duration": "1.5个月",
                    "tasks": [
                        "在多个数据集上进行测试",
                        "优化算法性能",
                        "撰写技术报告"
                    ]
                }
            ],
            "resource_requirements": [
                "多语言数据集（至少5种语言）",
                "GPU计算资源（至少4块V100）",
                "领域专家咨询（医疗、金融等）"
            ],
            "risk_assessment": [
                "数据获取困难：与数据平台建立合作关系，使用公开数据集",
                "算法性能不足：准备多种备选方案，预留调优时间",
                "计算资源不足：申请云计算资源，优化代码效率"
            ]
        }

# ================= MCP 工具注册 =================
# 创建全局系统实例
research_system = DeepResearchSystem(debug=True)

@mcp.tool()
async def create_research_project(name: str, description: str) -> Dict[str, Any]:
    """创建一个新的研究项目"""
    project = research_system.create_project(name, description)
    return project.to_dict()

@mcp.tool()
async def add_arxiv_paper_to_project(project_id: str, arxiv_url: str) -> Dict[str, Any]:
    """添加arXiv论文到研究项目"""
    # 提取arxiv ID
    arxiv_id = re.search(r'arxiv\.org/abs/([\d\.]+)', arxiv_url)
    if not arxiv_id:
        return {"error": "无效的arXiv URL"}
    
    arxiv_id = arxiv_id.group(1)
    paper = await research_system.get_arxiv_paper(arxiv_id)
    if not paper:
        return {"error": "无法获取论文信息"}
    
    research_system.save_paper(paper)
    research_system.add_paper_to_project(project_id, paper.paper_id)
    
    return {"status": "success", "paper_id": paper.paper_id}

@mcp.tool()
async def sync_zotero_with_project(project_id: str) -> Dict[str, Any]:
    """将Zotero文献库同步到研究项目"""
    return await research_system.sync_with_zotero(project_id)

@mcp.tool()
async def upload_pdf_to_project(project_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
    """上传PDF论文到研究项目"""
    paper = await research_system.upload_paper(file_content, filename)
    research_system.add_paper_to_project(project_id, paper.paper_id)
    return paper.to_dict()

@mcp.tool()
async def analyze_research_project(project_id: str) -> Dict[str, Any]:
    """分析研究项目并生成摘要"""
    summary = await research_system.analyze_project(project_id)
    return asdict(summary)

@mcp.tool()
async def generate_research_strategy_plan(project_id: str) -> Dict[str, Any]:
    """生成研究策略计划"""
    return await research_system.generate_research_strategy(project_id)

@mcp.tool()
async def visualize_citation_network(project_id: str, output_path: str = None) -> Dict[str, Any]:
    """可视化引文网络"""
    image_path = research_system.generate_citation_graph(project_id, output_path)
    return {"image_path": image_path}

@mcp.tool()
async def find_similar_papers(paper_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """查找相似论文"""
    papers = await research_system.find_similar_papers(paper_id, top_k)
    return [p.to_dict() for p in papers]

if __name__ == "__main__":
    # 初始化示例项目
    sample_project = research_system.create_project(
        "深度学习在NLP中的应用", 
        "研究深度学习技术在自然语言处理领域的最新进展和应用"
    )
    
    # 运行 FastMCP 服务器
    mcp.run()