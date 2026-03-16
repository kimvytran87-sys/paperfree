import csv
import html
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

USER_AGENT = "paperfree-search/1.0"
REQUEST_TIMEOUT = 30
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OUTPUT_CSV = "papers.csv"


@dataclass
class SearchConfig:
    core_keywords: str
    frontier_keywords: str
    start_date: str
    end_date: str
    max_per_source: int
    sources: List[str]


def split_keywords(text: str) -> List[str]:
    items = []
    for part in re.split(r"[\n,;]+", text or ""):
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            items.append(part)
    return items


def clean_abstract(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_iso_date(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    value = str(value).strip()
    datetime.strptime(value, "%Y-%m-%d")
    return value


def request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def request_text(url: str, params: Dict[str, Any]) -> str:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


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
        "select": "DOI,title,author,issued,URL,abstract,container-title,type"
    }
    data = request_json(CROSSREF_API, params)
    items = (data.get("message") or {}).get("items") or []
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
    xml_text = request_text(ARXIV_API, params)
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
    params = {
        "query": query,
        "format": "json",
        "pageSize": cfg.max_per_source,
        "resultType": "core",
    }
    data = request_json(EUROPEPMC_API, params)
    items = ((data.get("resultList") or {}).get("result")) or []
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


@st.dialog("Search settings")
def search_settings_dialog():
    st.write("Enter English keywords and an exact date range.")

    core = st.text_area(
        "Core keywords",
        value=st.session_state.get("core_keywords", "Urban Air Mobility, Advanced Air Mobility, UAM, AAM, eVTOL"),
        height=120,
        help="One per line or separated by commas.",
    )
    frontier = st.text_area(
        "Frontier keywords",
        value=st.session_state.get("frontier_keywords", "review, survey, public acceptance, urban planning, noise, environmental impact"),
        height=140,
        help="One per line or separated by commas.",
    )
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start date",
            value=st.session_state.get("start_date_obj", date(2020, 1, 1)),
            format="YYYY-MM-DD",
        )
    with c2:
        end_date = st.date_input(
            "End date",
            value=st.session_state.get("end_date_obj", date.today()),
            format="YYYY-MM-DD",
        )

    max_per_source = st.number_input(
        "Max results per source",
        min_value=10,
        max_value=300,
        value=int(st.session_state.get("max_per_source", 50)),
        step=10,
    )
    sources = st.multiselect(
        "Sources",
        ["crossref", "arxiv", "europepmc"],
        default=st.session_state.get("sources", ["crossref", "arxiv", "europepmc"]),
    )

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


def run_search(cfg: SearchConfig) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Searching sources...")
    active = cfg.sources or []
    total = len(active) if active else 1
    done = 0

    if "crossref" in active:
        try:
            rows.extend(search_crossref(cfg))
            st.success("Crossref search completed")
        except Exception as e:
            st.warning(f"Crossref search failed: {e}")
        done += 1
        progress.progress(done / total, text="Searching sources...")

    if "arxiv" in active:
        try:
            rows.extend(search_arxiv(cfg))
            st.success("arXiv search completed")
        except Exception as e:
            st.warning(f"arXiv search failed: {e}")
        done += 1
        progress.progress(done / total, text="Searching sources...")

    if "europepmc" in active:
        try:
            rows.extend(search_europepmc(cfg))
            st.success("Europe PMC search completed")
        except Exception as e:
            st.warning(f"Europe PMC search failed: {e}")
        done += 1
        progress.progress(done / total, text="Searching sources...")

    progress.empty()
    return deduplicate_rows(rows)


def save_csv(df: pd.DataFrame, path: str = OUTPUT_CSV):
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(page_title="Generate papers.csv", page_icon="📄", layout="wide")
    init_state()

    st.title("Generate papers.csv")
    st.caption("English keyword search + exact date filtering")

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

    if st.button("Search and generate papers.csv", type="primary"):
        cfg = SearchConfig(
            core_keywords=st.session_state["core_keywords"],
            frontier_keywords=st.session_state["frontier_keywords"],
            start_date=st.session_state["start_date"],
            end_date=st.session_state["end_date"],
            max_per_source=int(st.session_state["max_per_source"]),
            sources=st.session_state["sources"],
        )
        df = run_search(cfg)
        st.session_state["results_df"] = df
        save_csv(df, OUTPUT_CSV)
        st.success(f"papers.csv generated successfully. Records: {len(df)}")

    df = st.session_state.get("results_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Preview")
        st.dataframe(df, use_container_width=True, height=500)
        with open(OUTPUT_CSV, "rb") as f:
            st.download_button("Download papers.csv", data=f.read(), file_name="papers.csv", mime="text/csv")
    elif isinstance(df, pd.DataFrame) and df.empty:
        st.info("No records found in the selected date range.")


if __name__ == "__main__":
    main()
