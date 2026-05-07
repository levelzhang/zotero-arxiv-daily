from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
from datetime import datetime, timedelta, timezone

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
    # 更稳健的客户端配置：小页尺寸 + 适当重试与间隔
    client = arxiv.Client(page_size=25, num_retries=5, delay_seconds=3)

    categories = self.config.source.arxiv.category
    include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
    days = int(self.config.source.arxiv.get("days", 1))

    # 使用 UTC 日期，查询到昨天，避免当天/未来边界
    end_date = (datetime.now(timezone.utc).date() - timedelta(days=1))
    start_date = end_date - timedelta(days=max(days, 1))
    date_query = f"submittedDate:[{start_date.isoformat()} TO {end_date.isoformat()}]"

    # 类别查询与交叉列过滤（去掉 NOT 的多余括号）
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    if include_cross_list:
        cat_query = f"({cat_query})"
    else:
        cat_query = f"({cat_query}) AND NOT cross_list_cat:*"

    query_str = f"{cat_query} AND {date_query}"

    def run_search(q: str, attempts=4):
        for i in range(attempts):
            try:
                search = arxiv.Search(
                    query=q,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )
                return list(client.results(search))
            except arxiv.HTTPError:
                # 指数退避
                time.sleep(2 ** i)
        # 最终失败抛出
        raise

    # 主查询
    try:
        raw = run_search(query_str)
    except arxiv.HTTPError:
        # 回退1：移除 cross_list 过滤
        base_cat = " OR ".join(f"cat:{c}" for c in categories)
        query_no_cross = f"({base_cat}) AND {date_query}"
        try:
            raw = run_search(query_no_cross)
        except arxiv.HTTPError:
            # 回退2：缩短日期窗口到 7 天
            s2 = end_date - timedelta(days=7)
            query_short = f"({base_cat}) AND submittedDate:[{s2.isoformat()} TO {end_date.isoformat()}]"
            try:
                raw = run_search(query_short)
            except arxiv.HTTPError:
                # 回退3：移除日期过滤，抓取后本地按日期筛选
                query_latest = f"({base_cat})"
                raw = run_search(query_latest)
                def in_range(p):
                    d = getattr(p, "published", None) or getattr(p, "updated", None)
                    return d and (start_date <= d.date() <= end_date)
                raw = [p for p in raw if in_range(p)]

    # debug 截断
    if getattr(self.config.executor, "debug", False):
        raw = raw[:10]

    return raw

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
