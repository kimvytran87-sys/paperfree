import os
import re
import time
import json
import math
import html
import argparse
import traceback
from difflib import SequenceMatcher
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET


# =========================
# 基础配置
# =========================
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

USER_AGENT = "paperfree/0.1 (mailto:your_email@example.com)"
REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 1.0

DEFAULT_INCLUDE_KEYWORDS = [
    "urban air mobility",
    "advanced air mobility",
    "uam",
    "aam",
    "evtol",
    "review",
    "overview",
    "survey",
    "state of the art",
    "public acceptance",
    "social acceptance",
    "public perception",
    "social license",
    "urban planning",
    "built environment",
    "spatial planning",
    "infrastructure integration",
    "noise",
    "acoustic",
    "externalities",
    "environmental impact"
]

DEFAULT_EXCLUDE_KEYWORDS = [
    "editorial",
    "correction",
    "erratum",
    "book review",
    "conference committee",
    "call for papers"
]

# =========================
# 通用工具
# =========================
def safe_get(dct, path, default=None):
    cur = dct
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def clean_abstract(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def filename_safe(name: str, max_len: int = 150) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rstrip(" .")
    return name or "paper"


def keyword_score(text: str, include_keywords, exclude_keywords):
    text_norm = normalize_text(text)
    include_hits = []
    exclude_hits = []

    for kw in include_keywords:
        kw_norm = normalize_text(kw)
        if kw_norm and kw_norm in text_norm:
            include_hits.append(kw)

    for kw in exclude_keywords:
        kw_norm = normalize_text(kw)
        if kw_norm and kw_norm in text_norm:
            exclude_hits.append(kw)

    score = len(include_hits) * 2 - len(exclude_hits) * 3
    return score, include_hits, exclude_hits


def should_keep(score: int, min_score: int = 1) -> bool:
    return score >= min_score


def request_json(url: str, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    headers = headers or {}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def request_text(url: str, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    headers = headers or {}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text, None
    except Exception as e:
        return None, str(e)


# =========================
# Crossref 查询
# =========================
def search_crossref_by_title(title: str, mailto: str = ""):
    params = {
        "query.title": title,
        "rows": 5,
        "select": "DOI,title,author,issued,URL,abstract,container-title,type,is-referenced-by-count,license"
    }
    headers = {"User-Agent": USER_AGENT}
    if mailto:
        params["mailto"] = mailto

    data, err = request_json(CROSSREF_API, params=params, headers=headers)
    if err:
        return None, err

    items = safe_get(data, ["message", "items"], []) or []
    best = None
    best_sim = -1.0

    for item in items:
        item_title = ""
        if isinstance(item.get("title"), list) and item["title"]:
            item_title = item["title"][0]
        sim = title_similarity(title, item_title)
        if sim > best_sim:
            best_sim = sim
            best = item

    if best is None:
        return None, None

    result = {
        "source": "crossref",
        "matched_title": best.get("title", [""])[0] if best.get("title") else "",
        "similarity": round(best_sim, 4),
        "doi": best.get("DOI", ""),
        "abstract": clean_abstract(best.get("abstract", "")),
        "journal": best.get("container-title", [""])[0] if best.get("container-title") else "",
        "year": safe_get(best, ["issued", "date-parts", 0, 0], ""),
        "type": best.get("type", ""),
        "citation_count": best.get("is-referenced-by-count", 0),
        "landing_url": best.get("URL", ""),
        "pdf_url": "",
        "authors": parse_crossref_authors(best.get("author", [])),
        "license_url": parse_crossref_license(best.get("license", [])),
    }
    return result, None


def parse_crossref_authors(author_list):
    names = []
    for a in author_list or []:
        given = a.get("given", "") or ""
        family = a.get("family", "") or ""
        name = (given + " " + family).strip()
        if name:
            names.append(name)
    return "; ".join(names)


def parse_crossref_license(license_list):
    if not license_list:
        return ""
    first = license_list[0]
    return first.get("URL", "") or ""


# =========================
# arXiv 查询
# =========================
def search_arxiv_by_title(title: str):
    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 5,
    }
    headers = {"User-Agent": USER_AGENT}

    text, err = request_text(ARXIV_API, params=params, headers=headers)
    if err:
        return None, err

    try:
        root = ET.fromstring(text)
    except Exception as e:
        return None, f"解析 arXiv XML 失败: {e}"

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    entries = root.findall("atom:entry", ns)
    if not entries:
        return None, None

    best = None
    best_sim = -1.0

    for entry in entries:
        item_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        sim = title_similarity(title, item_title)
        if sim > best_sim:
            best_sim = sim
            best = entry

    if best is None:
        return None, None

    entry_id = best.findtext("atom:id", default="", namespaces=ns) or ""
    summary = best.findtext("atom:summary", default="", namespaces=ns) or ""
    published = best.findtext("atom:published", default="", namespaces=ns) or ""
    authors = []
    for author in best.findall("atom:author", ns):
        name = author.findtext("atom:name", default="", namespaces=ns) or ""
        if name:
            authors.append(name)

    pdf_url = ""
    landing_url = entry_id
    for link in best.findall("atom:link", ns):
        title_attr = link.attrib.get("title", "")
        href = link.attrib.get("href", "")
        if title_attr == "pdf":
            pdf_url = href
            break

    if not pdf_url and entry_id:
        pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

    arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""

    result = {
        "source": "arxiv",
        "matched_title": (best.findtext("atom:title", default="", namespaces=ns) or "").strip(),
        "similarity": round(best_sim, 4),
        "doi": "",
        "abstract": clean_abstract(summary),
        "journal": "arXiv",
        "year": published[:4] if published else "",
        "type": "preprint",
        "citation_count": "",
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "authors": "; ".join(authors),
        "license_url": "",
        "arxiv_id": arxiv_id,
    }
    return result, None


# =========================
# Europe PMC 查询
# =========================
def search_europepmc_by_title(title: str):
    query = f'TITLE:"{title}"'
    params = {
        "query": query,
        "format": "json",
        "pageSize": 5,
        "resultType": "core"
    }
    headers = {"User-Agent": USER_AGENT}

    data, err = request_json(EUROPEPMC_API, params=params, headers=headers)
    if err:
        return None, err

    results = safe_get(data, ["resultList", "result"], []) or []
    if not results:
        return None, None

    best = None
    best_sim = -1.0

    for item in results:
        item_title = item.get("title", "") or ""
        sim = title_similarity(title, item_title)
        if sim > best_sim:
            best_sim = sim
            best = item

    if best is None:
        return None, None

    pdf_url = ""
    fulltext_url_list = best.get("fullTextUrlList", {})
    fulltext_urls = fulltext_url_list.get("fullTextUrl", []) if isinstance(fulltext_url_list, dict) else []

    if isinstance(fulltext_urls, dict):
        fulltext_urls = [fulltext_urls]

    for ft in fulltext_urls:
        document_style = (ft.get("documentStyle", "") or "").lower()
        availability = (ft.get("availability", "") or "").lower()
        url = ft.get("url", "") or ""
        if document_style == "pdf" and url:
            pdf_url = url
            break
        if availability == "open access" and url and not pdf_url:
            pdf_url = url

    authors = best.get("authorString", "") or ""
    abstract = best.get("abstractText", "") or ""

    doi = best.get("doi", "") or ""
    psource = best.get("journalTitle", "") or ""
    pub_year = best.get("pubYear", "") or ""
    landing_url = best.get("fullTextUrl", "") or ""
    pmcid = best.get("pmcid", "") or ""

    if not landing_url and pmcid:
        landing_url = f"https://europepmc.org/article/PMC/{pmcid}"

    result = {
        "source": "europepmc",
        "matched_title": best.get("title", "") or "",
        "similarity": round(best_sim, 4),
        "doi": doi,
        "abstract": clean_abstract(abstract),
        "journal": psource,
        "year": pub_year,
        "type": best.get("pubType", ""),
        "citation_count": best.get("citedByCount", ""),
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "authors": authors,
        "license_url": "",
        "pmcid": pmcid,
    }
    return result, None


# =========================
# 来源融合
# =========================
def choose_best_record(title: str, crossref_rec, arxiv_rec, epmc_rec):
    candidates = [r for r in [crossref_rec, arxiv_rec, epmc_rec] if r]

    if not candidates:
        return None

    # 优先级：
    # 1. 相似度高
    # 2. 有 pdf_url
    # 3. 有 abstract
    # 4. 有 DOI
    def sort_key(r):
        return (
            float(r.get("similarity", 0)),
            1 if r.get("pdf_url") else 0,
            1 if r.get("abstract") else 0,
            1 if r.get("doi") else 0,
        )

    candidates.sort(key=sort_key, reverse=True)
    best = candidates[0].copy()

    # 尝试从其他来源补字段
    for extra in candidates[1:]:
        for key in ["doi", "abstract", "journal", "year", "authors", "landing_url", "pdf_url", "license_url"]:
            if not best.get(key) and extra.get(key):
                best[key] = extra[key]

    best["input_title"] = title
    return best


# =========================
# 下载 PDF
# =========================
def download_pdf(pdf_url: str, save_path: str, landing_url: str = "", doi: str = ""):
    headers = {"User-Agent": USER_AGENT}

    candidate_urls = []

    if pdf_url:
        candidate_urls.append(pdf_url)

    if landing_url:
        candidate_urls.append(landing_url)

    if doi:
        doi = doi.strip()
        if doi:
            candidate_urls.append(f"https://doi.org/{doi}")

    tried = []

    for url in candidate_urls:
        if not url:
            continue

        try:
            tried.append(url)

            # 先正常请求，允许跳转
            resp = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True,
            )
            resp.raise_for_status()

            final_url = resp.url or url
            content_type = (resp.headers.get("Content-Type", "") or "").lower()

            # 情况1：直接就是 PDF
            if "pdf" in content_type or final_url.lower().endswith(".pdf"):
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    return True, ""
                else:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    continue

            # 情况2：返回的是 HTML 页面，尝试从页面里找 PDF 链接
            if "html" in content_type or "text/" in content_type:
                try:
                    html_text = resp.text
                except Exception:
                    html_text = ""

                pdf_patterns = [
                    r'https?://[^\s"\']+\.pdf',
                    r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
                    r'href=["\']([^"\']*/pdf/[^"\']*)["\']',
                    r'href=["\']([^"\']*download[^"\']*pdf[^"\']*)["\']',
                ]

                found_links = []
                for pattern in pdf_patterns:
                    matches = re.findall(pattern, html_text, flags=re.IGNORECASE)
                    for m in matches:
                        if isinstance(m, tuple):
                            m = m[0]
                        if m:
                            found_links.append(m)

                # 相对链接补全
                normalized_links = []
                for link in found_links:
                    if link.startswith("http://") or link.startswith("https://"):
                        normalized_links.append(link)
                    elif link.startswith("/"):
                        from urllib.parse import urlparse
                        parsed = urlparse(final_url)
                        normalized_links.append(f"{parsed.scheme}://{parsed.netloc}{link}")

                # 去重
                seen = set()
                unique_links = []
                for link in normalized_links:
                    if link not in seen:
                        seen.add(link)
                        unique_links.append(link)

                # 尝试下载找到的 PDF 链接
                for pdf_link in unique_links:
                    try:
                        pdf_resp = requests.get(
                            pdf_link,
                            headers=headers,
                            timeout=REQUEST_TIMEOUT,
                            stream=True,
                            allow_redirects=True,
                        )
                        pdf_resp.raise_for_status()

                        pdf_content_type = (pdf_resp.headers.get("Content-Type", "") or "").lower()
                        if "pdf" in pdf_content_type or pdf_resp.url.lower().endswith(".pdf"):
                            with open(save_path, "wb") as f:
                                for chunk in pdf_resp.iter_content(chunk_size=1024 * 64):
                                    if chunk:
                                        f.write(chunk)

                            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                                return True, ""
                            else:
                                if os.path.exists(save_path):
                                    os.remove(save_path)
                    except Exception:
                        continue

        except Exception:
            continue

    return False, f"下载失败，已尝试: {' | '.join(tried)}"


# =========================
# 主流程
# =========================
def process_one_title(title, include_keywords, exclude_keywords, pdf_dir, mailto, similarity_threshold):
    result = {
        "input_title": title,
        "matched_title": "",
        "source": "",
        "similarity": 0,
        "doi": "",
        "journal": "",
        "year": "",
        "authors": "",
        "abstract": "",
        "landing_url": "",
        "pdf_url": "",
        "license_url": "",
        "type": "",
        "citation_count": "",
        "include_hits": "",
        "exclude_hits": "",
        "keyword_score": "",
        "keep": "",
        "downloaded_pdf": "",
        "pdf_path": "",
        "status": "",
        "error": "",
    }

    try:
        crossref_rec, crossref_err = search_crossref_by_title(title, mailto=mailto)
        time.sleep(SLEEP_SECONDS)

        arxiv_rec, arxiv_err = search_arxiv_by_title(title)
        time.sleep(SLEEP_SECONDS)

        epmc_rec, epmc_err = search_europepmc_by_title(title)
        time.sleep(SLEEP_SECONDS)

        best = choose_best_record(title, crossref_rec, arxiv_rec, epmc_rec)

        if not best:
            result["status"] = "not_found"
            result["error"] = "三个免费来源都未找到足够匹配的记录"
            return result

        if float(best.get("similarity", 0)) < similarity_threshold:
            result["status"] = "low_similarity"
            result["matched_title"] = best.get("matched_title", "")
            result["source"] = best.get("source", "")
            result["similarity"] = best.get("similarity", 0)
            result["error"] = f"匹配度低于阈值 {similarity_threshold}"
            return result

        for k in ["matched_title", "source", "similarity", "doi", "journal", "year", "authors",
                  "abstract", "landing_url", "pdf_url", "license_url", "type", "citation_count"]:
            result[k] = best.get(k, "")

        text_for_screen = f"{best.get('matched_title', '')} {best.get('abstract', '')} {best.get('journal', '')}"
        score, include_hits, exclude_hits = keyword_score(
            text_for_screen,
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords
        )
        result["include_hits"] = "; ".join(include_hits)
        result["exclude_hits"] = "; ".join(exclude_hits)
        result["keyword_score"] = score
        result["keep"] = should_keep(score)

        # 下载开放获取 PDF
        pdf_url = best.get("pdf_url", "")
        if pdf_url:
            save_name = filename_safe(best.get("matched_title") or title) + ".pdf"
            save_path = os.path.join(pdf_dir, save_name)
            ok, msg = download_pdf(
                pdf_url=pdf_url,
                save_path=save_path,
                landing_url=best.get("landing_url", ""),
                doi=best.get("doi", "")
            )
            if ok:
                result["downloaded_pdf"] = True
                result["pdf_path"] = save_path
            else:
                result["downloaded_pdf"] = False
                result["error"] = msg
        else:
            result["downloaded_pdf"] = False

        result["status"] = "ok"
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{e}\n{traceback.format_exc()}"
        return result


def load_titles_from_csv(csv_path: str, title_col: str = "title"):
    df = pd.read_csv(csv_path)
    if title_col not in df.columns:
        raise ValueError(f"CSV 中没有找到列名 '{title_col}'。当前列名为: {list(df.columns)}")

    titles = []
    for val in df[title_col].tolist():
        if pd.isna(val):
            continue
        title = str(val).strip()
        if title:
            titles.append(title)
    return titles


def save_outputs(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    all_path = os.path.join(out_dir, "results_all.csv")
    keep_path = os.path.join(out_dir, "results_keep.csv")
    review_path = os.path.join(out_dir, "results_review.csv")
    summary_path = os.path.join(out_dir, "summary.json")

    df.to_csv(all_path, index=False, encoding="utf-8-sig")

    df_keep = df[df["keep"] == True].copy()
    df_keep.to_csv(keep_path, index=False, encoding="utf-8-sig")

    df_review = df[
        (df["status"].isin(["low_similarity", "not_found", "error"])) |
        ((df["keep"] == False) & (df["status"] == "ok"))
    ].copy()
    df_review.to_csv(review_path, index=False, encoding="utf-8-sig")

    summary = {
        "total": int(len(df)),
        "ok": int((df["status"] == "ok").sum()),
        "keep_true": int((df["keep"] == True).sum()),
        "downloaded_pdf_true": int((df["downloaded_pdf"] == True).sum()),
        "not_found": int((df["status"] == "not_found").sum()),
        "low_similarity": int((df["status"] == "low_similarity").sum()),
        "error": int((df["status"] == "error").sum()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return all_path, keep_path, review_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="免费批量查论文 + 下载开放获取 PDF + 粗筛摘要")
    parser.add_argument("--input", default="papers.csv", help="输入 CSV 文件路径，默认 papers.csv")
    parser.add_argument("--title-col", default="title", help="标题列名，默认 title")
    parser.add_argument("--output-dir", default="output", help="输出目录，默认 output")
    parser.add_argument("--pdf-dir", default="output/pdfs", help="PDF 保存目录，默认 output/pdfs")
    parser.add_argument("--mailto", default="", help="你的邮箱，可选；用于 Crossref polite pool")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="标题匹配阈值，默认 0.75")
    parser.add_argument("--include-keywords", default=",".join(DEFAULT_INCLUDE_KEYWORDS),
                        help="包含关键词，逗号分隔")
    parser.add_argument("--exclude-keywords", default=",".join(DEFAULT_EXCLUDE_KEYWORDS),
                        help="排除关键词，逗号分隔")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.pdf_dir)

    include_keywords = [x.strip() for x in args.include_keywords.split(",") if x.strip()]
    exclude_keywords = [x.strip() for x in args.exclude_keywords.split(",") if x.strip()]

    print("=" * 70)
    print("开始处理论文列表")
    print(f"输入文件: {args.input}")
    print(f"标题列名: {args.title_col}")
    print(f"输出目录: {args.output_dir}")
    print(f"PDF目录 : {args.pdf_dir}")
    print("=" * 70)

    titles = load_titles_from_csv(args.input, args.title_col)
    print(f"共读取到 {len(titles)} 个标题")

    rows = []
    for title in tqdm(titles, desc="处理进度"):
        row = process_one_title(
            title=title,
            include_keywords=include_keywords,
            exclude_keywords=exclude_keywords,
            pdf_dir=args.pdf_dir,
            mailto=args.mailto,
            similarity_threshold=args.similarity_threshold
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    all_path, keep_path, review_path, summary_path = save_outputs(df, args.output_dir)

    print("\n处理完成。输出文件如下：")
    print(f"1. 全部结果: {all_path}")
    print(f"2. 建议保留: {keep_path}")
    print(f"3. 建议复核: {review_path}")
    print(f"4. 统计摘要: {summary_path}")

    print("\n说明：")
    print("- results_all.csv：所有标题的匹配、摘要、DOI、PDF 下载结果")
    print("- results_keep.csv：按关键词粗筛后建议保留的论文")
    print("- results_review.csv：未找到、低匹配、出错、或粗筛未通过的论文")
    print("- output/pdfs/：成功下载的开放获取 PDF")

    if len(df) > 0:
        print("\n前 5 条结果预览：")
        preview_cols = ["input_title", "matched_title", "source", "similarity", "doi",
                        "journal", "year", "keep", "downloaded_pdf", "status"]
        print(df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()