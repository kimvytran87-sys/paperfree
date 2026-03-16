import csv
import html
import json
import os
import re
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

# =========================
# Config
# =========================
SEARCH_USER_AGENT = "paperfree-search/2.0"
USER_AGENT = "paperfree/2.0"
REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 1.0
OUTPUT_CSV = "papers.csv"

CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

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
    "environmental impact",
]

DEFAULT_EXCLUDE_KEYWORDS = [
    "editorial",
    "correction",
    "erratum",
    "book review",
    "conference committee",
    "call for papers",
]

METHOD_PATTERNS = {
    "systematic literature review": [r"\bsystematic literature review\b", r"\bsystematic review\b"],
    "scoping review": [r"\bscoping review\b"],
    "bibliometric analysis": [r"\bbibliometric\b", r"\bcitation analysis\b"],
    "meta-analysis": [r"\bmeta-analysis\b", r"\bmeta analysis\b"],
    "survey": [r"\bsurvey\b"],
    "case study": [r"\bcase study\b", r"\bcase studies\b"],
    "simulation": [r"\bsimulation\b", r"\bsimulated\b"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b"],
    "mixed methods": [r"\bmixed methods\b"],
    "interview": [r"\binterview\b", r"\binterviews\b"],
    "questionnaire": [r"\bquestionnaire\b"],
    "agent-based model": [r"\bagent-based model\b", r"\bagent based model\b"],
    "gis": [r"\bgis\b", r"\bgeographic information system\b"],
    "machine learning": [r"\bmachine learning\b"],
    "deep learning": [r"\bdeep learning\b"],
    "discrete choice model": [r"\bdiscrete choice\b", r"\blogit model\b"],
}

MODEL_PATTERNS = {
    "TAM": [r"\btechnology acceptance model\b", r"\btam\b"],
    "UTAUT": [r"\butaut\b", r"\bunified theory of acceptance and use of technology\b"],
    "TPB": [r"\btheory of planned behavior\b", r"\btpb\b"],
    "SEM": [r"\bstructural equation model\b", r"\bstructural equation modeling\b", r"\bsem\b"],
    "CFD": [r"\bcfd\b", r"\bcomputational fluid dynamics\b"],
    "AHP": [r"\bahp\b", r"\banalytic hierarchy process\b"],
    "TOPSIS": [r"\btopsis\b"],
    "LCA": [r"\blife cycle assessment\b", r"\blca\b"],
    "ANN": [r"\bartificial neural network\b", r"\bann\b"],
    "SVM": [r"\bsupport vector machine\b", r"\bsvm\b"],
    "BERT": [r"\bbert\b"],
    "LSTM": [r"\blstm\b"],
}


@dataclass
class SearchConfig:
    core_keywords: str
    frontier_keywords: str
    start_date: str
    end_date: str
    max_per_source: int
    sources: List[str]


# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_get(dct, path, default=None):
    cur = dct
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def split_keywords(text: str) -> List[str]:
    items = []
    for part in re.split(r"[\n,;]+", text or ""):
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            items.append(part)
    return items


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def clean_abstract(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def filename_safe(name: str, max_len: int = 150) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rstrip(" .")
    return name or "paper"


def ensure_iso_date(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    value = str(value).strip()
    datetime.strptime(value, "%Y-%m-%d")
    return value


def date_in_range(pub_date: str, start_date: str, end_date: str) -> bool:
    if not pub_date:
        return True
    try:
        d = datetime.strptime(pub_date[:10], "%Y-%m-%d").date()
        s = datetime.strptime(start_date, "%Y-%m-%d").date()
        e = datetime.strptime(end_date, "%Y-%m-%d").date()
        return s <= d <= e
    except Exception:
        return True


def request_json(url: str, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    headers = headers or {}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def request_text(url: str, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    headers = headers or {}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


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


def extract_items(text: str, pattern_dict: Dict[str, List[str]]) -> List[str]:
    text_norm = normalize_text(text)
    found = []
    for label, patterns in pattern_dict.items():
        for pattern in patterns:
            if re.search(pattern, text_norm, flags=re.IGNORECASE):
                found.append(label)
                break
    return found


def generate_brief_summary(title: str, abstract: str, methods: List[str], models: List[str]) -> str:
    abs_clean = clean_abstract(abstract)
    if not abs_clean:
        return f"{title}. No abstract available."
    first = re.split(r"(?<=[.!?])\s+", abs_clean)
    first_sentence = first[0].strip() if first else abs_clean[:300].strip()
    tail = []
    if methods:
        tail.append("Methods: " + ", ".join(methods[:3]))
    if models:
        tail.append("Models: " + ", ".join(models[:3]))
    return first_sentence + (" | " + " ; ".join(tail) if tail else "")


# =========================
# Search stage
# =========================
def build_query(core_keywords: str, frontier_keywords: str) -> str:
    core = split_keywords(core_keywords)
    frontier = split_keywords(frontier_keywords)
    core_block = " OR ".join([f'"{x}"' for x in core])
    frontier_block = " OR ".join([f'"{x}"' for x in frontier])
    if core_block and frontier_block:
        return f"({core_block}) AND ({frontier_block})"
    if core_block:
        return core_block
    return frontier_block


def parse_crossref_date(item: Dict[str, Any]) -> str:
    date_parts = (item.get("issued") or {}).get("date-parts") or []
    if date_parts and date_parts[0]:
        parts = date_parts[0]
        y = str(parts[0]) if len(parts) >= 1 else ""
        m = f"{int(parts[1]):02d}" if len(parts) >= 2 else "01"
        d = f"{int(parts[2]):02d}" if len(parts) >= 3 else "01"
        if y:
            return f"{y}-{m}-{d}"
    return ""


def search_crossref(cfg: SearchConfig) -> List[Dict[str, Any]]:
    query_text = build_query(cfg.core_keywords, cfg.frontier_keywords)
    params = {
        "query": query_text,
        "rows": cfg.max_per_source,
        "select": "DOI,title,author,issued,URL,abstract,container-title,type",
    }
    data = request_json(CROSSREF_API, params=params, headers={"User-Agent": SEARCH_USER_AGENT})
    items = safe_get(data, ["message", "items"], []) or []
    rows = []
    for item in items:
        pub_date = parse_crossref_date(item)
        if not date_in_range(pub_date, cfg.start_date, cfg.end_date):
            continue
        title = ((item.get("title") or [""])[0] or "").strip()
        authors = []
        for a in item.get("author") or []:
            name = f"{a.get('given','')} {a.get('family','')}".strip()
            if name:
                authors.append(name)
        rows.append({
            "title": title,
            "abstract": clean_abstract(item.get("abstract", "")),
            "doi": item.get("DOI", ""),
            "year": pub_date[:4] if pub_date else "",
            "pub_date": pub_date,
            "source": "crossref",
            "authors": "; ".join(authors),
            "journal": ((item.get("container-title") or [""])[0] or "").strip(),
            "url": item.get("URL", ""),
            "pdf_url": "",
            "type": item.get("type", ""),
            "query": query_text,
        })
    return rows


def search_arxiv(cfg: SearchConfig) -> List[Dict[str, Any]]:
    query_text = build_query(cfg.core_keywords, cfg.frontier_keywords)
    params = {
        "search_query": f"all:{query_text}",
        "start": 0,
        "max_results": cfg.max_per_source,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    xml_text = request_text(ARXIV_API, params=params, headers={"User-Agent": SEARCH_USER_AGENT})
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
        if not date_in_range(published, cfg.start_date, cfg.end_date):
            continue
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        authors = []
        for author in entry.findall("atom:author", ns):
            name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
        rows.append({
            "title": title,
            "abstract": clean_abstract(summary),
            "doi": "",
            "year": published[:4] if published else "",
            "pub_date": published,
            "source": "arxiv",
            "authors": "; ".join(authors),
            "journal": "arXiv",
            "url": entry_id,
            "pdf_url": pdf_url,
            "type": "preprint",
            "query": query_text,
        })
    return rows


def search_europepmc(cfg: SearchConfig) -> List[Dict[str, Any]]:
    query_text = build_query(cfg.core_keywords, cfg.frontier_keywords)
    query = f'({query_text}) AND FIRST_PDATE:[{cfg.start_date} TO {cfg.end_date}]'
    params = {"query": query, "format": "json", "pageSize": cfg.max_per_source, "resultType": "core"}
    data = request_json(EUROPEPMC_API, params=params, headers={"User-Agent": SEARCH_USER_AGENT})
    items = safe_get(data, ["resultList", "result"], []) or []
    rows = []
    for item in items:
        pub_date = (item.get("firstPublicationDate") or item.get("electronicPublicationDate") or item.get("pubYear") or "").strip()
        if len(pub_date) == 4:
            pub_date = f"{pub_date}-01-01"
        if pub_date and not date_in_range(pub_date, cfg.start_date, cfg.end_date):
            continue
        pdf_url = ""
        ft = item.get("fullTextUrlList") or {}
        ft_list = ft.get("fullTextUrl") or []
        if isinstance(ft_list, dict):
            ft_list = [ft_list]
        for x in ft_list:
            url = x.get("url", "")
            style = (x.get("documentStyle") or "").lower()
            if style == "pdf" and url:
                pdf_url = url
                break
        rows.append({
            "title": (item.get("title") or "").strip(),
            "abstract": clean_abstract(item.get("abstractText", "")),
            "doi": item.get("doi", ""),
            "year": pub_date[:4] if pub_date else "",
            "pub_date": pub_date,
            "source": "europepmc",
            "authors": item.get("authorString", ""),
            "journal": item.get("journalTitle", ""),
            "url": item.get("fullTextUrl", "") or f"https://europepmc.org/article/MED/{item.get('id','')}",
            "pdf_url": pdf_url,
            "type": item.get("pubType", ""),
            "query": query_text,
        })
    return rows


def deduplicate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_key"] = df.apply(lambda r: (str(r.get("doi", "")).strip().lower() or str(r.get("title", "")).strip().lower()), axis=1)
    df = df.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"])
    order = ["title", "abstract", "doi", "year", "pub_date", "source", "authors", "journal", "url", "pdf_url", "type", "query"]
    for col in order:
        if col not in df.columns:
            df[col] = ""
    return df[order]


def save_csv(df: pd.DataFrame, path: str = OUTPUT_CSV):
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


# =========================
# Download/enrich stage
# =========================
def search_crossref_by_title(title: str, mailto: str = ""):
    params = {
        "query.title": title,
        "rows": 5,
        "select": "DOI,title,author,issued,URL,abstract,container-title,type,is-referenced-by-count,license",
    }
    headers = {"User-Agent": USER_AGENT}
    if mailto:
        params["mailto"] = mailto

    try:
        data = request_json(CROSSREF_API, params=params, headers=headers)
    except Exception as e:
        return None, str(e)

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


def search_arxiv_by_title(title: str):
    params = {"search_query": f'ti:"{title}"', "start": 0, "max_results": 5}
    try:
        text = request_text(ARXIV_API, params=params, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        return None, str(e)
    try:
        root = ET.fromstring(text)
    except Exception as e:
        return None, f"解析 arXiv XML 失败: {e}"
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
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
        if link.attrib.get("title", "") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break
    if not pdf_url and entry_id:
        pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
    return {
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
    }, None


def search_europepmc_by_title(title: str):
    params = {"query": f'TITLE:"{title}"', "format": "json", "pageSize": 5, "resultType": "core"}
    try:
        data = request_json(EUROPEPMC_API, params=params, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        return None, str(e)
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
    landing_url = best.get("fullTextUrl", "") or ""
    pmcid = best.get("pmcid", "") or ""
    if not landing_url and pmcid:
        landing_url = f"https://europepmc.org/article/PMC/{pmcid}"
    return {
        "source": "europepmc",
        "matched_title": best.get("title", "") or "",
        "similarity": round(best_sim, 4),
        "doi": best.get("doi", "") or "",
        "abstract": clean_abstract(best.get("abstractText", "")),
        "journal": best.get("journalTitle", "") or "",
        "year": best.get("pubYear", "") or "",
        "type": best.get("pubType", ""),
        "citation_count": best.get("citedByCount", ""),
        "landing_url": landing_url,
        "pdf_url": pdf_url,
        "authors": best.get("authorString", "") or "",
        "license_url": "",
    }, None


def choose_best_record(title: str, crossref_rec, arxiv_rec, epmc_rec):
    candidates = [r for r in [crossref_rec, arxiv_rec, epmc_rec] if r]
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            float(r.get("similarity", 0)),
            1 if r.get("pdf_url") else 0,
            1 if r.get("abstract") else 0,
            1 if r.get("doi") else 0,
        ),
        reverse=True,
    )
    best = candidates[0].copy()
    for extra in candidates[1:]:
        for key in ["doi", "abstract", "journal", "year", "authors", "landing_url", "pdf_url", "license_url"]:
            if not best.get(key) and extra.get(key):
                best[key] = extra[key]
    best["input_title"] = title
    return best


def download_pdf(pdf_url: str, save_path: str, landing_url: str = "", doi: str = ""):
    headers = {"User-Agent": USER_AGENT}
    candidate_urls = []
    if pdf_url:
        candidate_urls.append(pdf_url)
    if landing_url:
        candidate_urls.append(landing_url)
    if doi:
        candidate_urls.append(f"https://doi.org/{doi.strip()}")
    tried = []
    for url in candidate_urls:
        if not url:
            continue
        try:
            tried.append(url)
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
            resp.raise_for_status()
            final_url = resp.url or url
            content_type = (resp.headers.get("Content-Type", "") or "").lower()
            if "pdf" in content_type or final_url.lower().endswith(".pdf"):
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                    return True, ""
                if os.path.exists(save_path):
                    os.remove(save_path)
                continue
            if "html" in content_type or "text/" in content_type:
                html_text = getattr(resp, "text", "") or ""
                pdf_patterns = [
                    r'https?://[^\s"\']+\.pdf',
                    r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
                    r'href=["\']([^"\']*/pdf/[^"\']*)["\']',
                    r'href=["\']([^"\']*download[^"\']*pdf[^"\']*)["\']',
                ]
                found_links = []
                for pattern in pdf_patterns:
                    for m in re.findall(pattern, html_text, flags=re.IGNORECASE):
                        found_links.append(m[0] if isinstance(m, tuple) else m)
                normalized_links = []
                for link in found_links:
                    if link.startswith(("http://", "https://")):
                        normalized_links.append(link)
                    elif link.startswith("/"):
                        parsed = urlparse(final_url)
                        normalized_links.append(f"{parsed.scheme}://{parsed.netloc}{link}")
                unique_links = list(dict.fromkeys(normalized_links))
                for pdf_link in unique_links:
                    try:
                        pdf_resp = requests.get(pdf_link, headers=headers, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
                        pdf_resp.raise_for_status()
                        pdf_content_type = (pdf_resp.headers.get("Content-Type", "") or "").lower()
                        if "pdf" in pdf_content_type or pdf_resp.url.lower().endswith(".pdf"):
                            with open(save_path, "wb") as f:
                                for chunk in pdf_resp.iter_content(chunk_size=1024 * 64):
                                    if chunk:
                                        f.write(chunk)
                            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                                return True, ""
                            if os.path.exists(save_path):
                                os.remove(save_path)
                    except Exception:
                        continue
        except Exception:
            continue
    return False, f"下载失败，已尝试: {' | '.join(tried)}"


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
        "methods": "",
        "models": "",
        "brief_summary": "",
    }
    try:
        crossref_rec, _ = search_crossref_by_title(title, mailto=mailto)
        time.sleep(SLEEP_SECONDS)
        arxiv_rec, _ = search_arxiv_by_title(title)
        time.sleep(SLEEP_SECONDS)
        epmc_rec, _ = search_europepmc_by_title(title)
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
        score, include_hits, exclude_hits = keyword_score(text_for_screen, include_keywords, exclude_keywords)
        result["include_hits"] = "; ".join(include_hits)
        result["exclude_hits"] = "; ".join(exclude_hits)
        result["keyword_score"] = score
        result["keep"] = should_keep(score)
        methods = extract_items(text_for_screen, METHOD_PATTERNS)
        models = extract_items(text_for_screen, MODEL_PATTERNS)
        result["methods"] = "; ".join(methods)
        result["models"] = "; ".join(models)
        result["brief_summary"] = generate_brief_summary(best.get("matched_title", title), best.get("abstract", ""), methods, models)
        save_name = filename_safe(best.get("matched_title") or title) + ".pdf"
        save_path = os.path.join(pdf_dir, save_name)
        ok, msg = download_pdf(best.get("pdf_url", ""), save_path, best.get("landing_url", ""), best.get("doi", ""))
        if ok:
            result["downloaded_pdf"] = True
            result["pdf_path"] = save_path
        else:
            result["downloaded_pdf"] = False
            result["error"] = msg
        result["status"] = "ok"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{e}\n{traceback.format_exc()}"
        return result


def save_outputs(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    all_path = os.path.join(out_dir, "results_all.csv")
    keep_path = os.path.join(out_dir, "results_keep.csv")
    review_path = os.path.join(out_dir, "results_review.csv")
    reading_path = os.path.join(out_dir, "literature_reading_table.csv")
    methods_path = os.path.join(out_dir, "methods_models_summary.csv")
    summary_path = os.path.join(out_dir, "summary.json")

    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    df[df["keep"] == True].copy().to_csv(keep_path, index=False, encoding="utf-8-sig")
    df[(df["status"].isin(["low_similarity", "not_found", "error"])) | ((df["keep"] == False) & (df["status"] == "ok"))].copy().to_csv(review_path, index=False, encoding="utf-8-sig")

    reading_cols = [
        "input_title", "matched_title", "year", "authors", "journal", "doi", "source",
        "brief_summary", "methods", "models", "abstract", "downloaded_pdf", "pdf_path", "status"
    ]
    reading_df = df.copy()
    for col in reading_cols:
        if col not in reading_df.columns:
            reading_df[col] = ""
    reading_df[reading_cols].to_csv(reading_path, index=False, encoding="utf-8-sig")

    methods_df = df[["matched_title", "methods", "models", "brief_summary", "doi", "journal", "year"]].copy()
    methods_df.to_csv(methods_path, index=False, encoding="utf-8-sig")

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
    return {
        "all": all_path,
        "keep": keep_path,
        "review": review_path,
        "reading": reading_path,
        "methods": methods_path,
        "summary": summary_path,
    }


def run_download_pipeline(input_csv: str, output_dir: str, pdf_dir: str, include_keywords: List[str], exclude_keywords: List[str], mailto: str = "", similarity_threshold: float = 0.75):
    df_input = pd.read_csv(input_csv)
    if "title" not in df_input.columns:
        raise ValueError(f"CSV 中没有找到列名 'title'。当前列名为: {list(df_input.columns)}")
    titles = [str(v).strip() for v in df_input["title"].tolist() if pd.notna(v) and str(v).strip()]
    rows = []
    progress = st.progress(0, text="Downloading and enriching papers...")
    total = max(len(titles), 1)
    for idx, title in enumerate(titles, 1):
        rows.append(process_one_title(title, include_keywords, exclude_keywords, pdf_dir, mailto, similarity_threshold))
        progress.progress(idx / total, text=f"Downloading and enriching papers... {idx}/{total}")
    progress.empty()
    return pd.DataFrame(rows)


# =========================
# Streamlit UI
# =========================
@st.dialog("Search settings")
def search_settings_dialog():
    st.write("Enter English keywords and an exact date range.")
    core = st.text_area("Core keywords", value=st.session_state.get("core_keywords", "Urban Air Mobility, Advanced Air Mobility, UAM, AAM, eVTOL"), height=120)
    frontier = st.text_area("Frontier keywords", value=st.session_state.get("frontier_keywords", "review, survey, public acceptance, urban planning, noise, environmental impact"), height=140)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=st.session_state.get("start_date_obj", date(2020, 1, 1)), format="YYYY-MM-DD")
    with c2:
        end_date = st.date_input("End date", value=st.session_state.get("end_date_obj", date.today()), format="YYYY-MM-DD")
    max_per_source = st.number_input("Max results per source", min_value=10, max_value=300, value=int(st.session_state.get("max_per_source", 50)), step=10)
    sources = st.multiselect("Sources", ["crossref", "arxiv", "europepmc"], default=st.session_state.get("sources", ["crossref", "arxiv", "europepmc"]))
    c3, c4 = st.columns(2)
    with c3:
        if st.button("Save", use_container_width=True):
            st.session_state["core_keywords"] = core.strip()
            st.session_state["frontier_keywords"] = frontier.strip()
            st.session_state["start_date"] = ensure_iso_date(start_date)
            st.session_state["end_date"] = ensure_iso_date(end_date)
            st.session_state["start_date_obj"] = start_date
            st.session_state["end_date_obj"] = end_date
            st.session_state["max_per_source"] = int(max_per_source)
            st.session_state["sources"] = sources
            st.rerun()
    with c4:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def init_state():
    defaults = {
        "core_keywords": "Urban Air Mobility, Advanced Air Mobility, UAM, AAM, eVTOL",
        "frontier_keywords": "review, survey, public acceptance, urban planning, noise, environmental impact",
        "start_date": "2020-01-01",
        "end_date": date.today().isoformat(),
        "start_date_obj": date(2020, 1, 1),
        "end_date_obj": date.today(),
        "max_per_source": 50,
        "sources": ["crossref", "arxiv", "europepmc"],
        "results_df": None,
        "download_df": None,
        "output_paths": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_search(cfg: SearchConfig) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Searching sources...")
    active = cfg.sources or []
    total = len(active) if active else 1
    done = 0
    for source in active:
        try:
            if source == "crossref":
                rows.extend(search_crossref(cfg))
                st.success("Crossref search completed")
            elif source == "arxiv":
                rows.extend(search_arxiv(cfg))
                st.success("arXiv search completed")
            elif source == "europepmc":
                rows.extend(search_europepmc(cfg))
                st.success("Europe PMC search completed")
        except Exception as e:
            st.warning(f"{source} search failed: {e}")
        done += 1
        progress.progress(done / total, text="Searching sources...")
    progress.empty()
    return deduplicate_rows(rows)


def main():
    st.set_page_config(page_title="Search -> papers.csv -> download", page_icon="📄", layout="wide")
    init_state()
    st.title("Integrated Paper Pipeline")
    st.caption("Search dialog → generate papers.csv → auto download → abstract + methods/models CSV")

    if st.button("Open search dialog"):
        search_settings_dialog()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Core keywords**")
        st.write(st.session_state["core_keywords"])
    with c2:
        st.write("**Frontier keywords**")
        st.write(st.session_state["frontier_keywords"])
    with c3:
        st.write("**Date range**")
        st.write(f"{st.session_state['start_date']} to {st.session_state['end_date']}")
        st.write(f"**Sources**: {', '.join(st.session_state['sources'])}")

    if st.button("Search + generate papers.csv + auto download", type="primary"):
        cfg = SearchConfig(
            core_keywords=st.session_state["core_keywords"],
            frontier_keywords=st.session_state["frontier_keywords"],
            start_date=st.session_state["start_date"],
            end_date=st.session_state["end_date"],
            max_per_source=int(st.session_state["max_per_source"]),
            sources=st.session_state["sources"],
        )
        search_df = run_search(cfg)
        st.session_state["results_df"] = search_df
        save_csv(search_df, OUTPUT_CSV)
        st.success(f"papers.csv generated successfully. Records: {len(search_df)}")

        include_keywords = split_keywords(st.session_state["core_keywords"]) + split_keywords(st.session_state["frontier_keywords"])
        exclude_keywords = DEFAULT_EXCLUDE_KEYWORDS
        output_dir = "output"
        pdf_dir = os.path.join(output_dir, "pdfs")
        ensure_dir(output_dir)
        ensure_dir(pdf_dir)

        download_df = run_download_pipeline(OUTPUT_CSV, output_dir, pdf_dir, include_keywords, exclude_keywords)
        st.session_state["download_df"] = download_df
        st.session_state["output_paths"] = save_outputs(download_df, output_dir)
        st.success("Download and CSV generation completed")

    search_df = st.session_state.get("results_df")
    if isinstance(search_df, pd.DataFrame) and not search_df.empty:
        st.subheader("Search results preview")
        st.dataframe(search_df, use_container_width=True, height=280)
        with open(OUTPUT_CSV, "rb") as f:
            st.download_button("Download papers.csv", data=f.read(), file_name="papers.csv", mime="text/csv")

    download_df = st.session_state.get("download_df")
    output_paths = st.session_state.get("output_paths") or {}
    if isinstance(download_df, pd.DataFrame) and not download_df.empty:
        st.subheader("Download and enrichment preview")
        preview_cols = ["input_title", "matched_title", "source", "doi", "downloaded_pdf", "methods", "models", "brief_summary", "status"]
        for col in preview_cols:
            if col not in download_df.columns:
                download_df[col] = ""
        st.dataframe(download_df[preview_cols], use_container_width=True, height=400)
        for label, path in output_paths.items():
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(f"Download {os.path.basename(path)}", data=f.read(), file_name=os.path.basename(path), mime="text/csv" if path.endswith(".csv") else "application/json", key=f"dl_{label}")
    elif isinstance(download_df, pd.DataFrame) and download_df.empty:
        st.info("No downloadable records were produced.")


if __name__ == "__main__":
    main()
