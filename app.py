import re
import sys
import os
import json
import textwrap
import unicodedata
import time
from loguru import logger
import google.genai as genai

import requests
import streamlit as st

# 프롬프트 로드 (modules/prompts.py 파일이 있다고 가정)
from modules.prompts import (
    BRAILLE_TO_KOREAN_SYSTEM_PROMPT,
    KOREAN_TO_BRAILLE_SYSTEM_PROMPT,
    BRAILLE_TO_CHINESE_SYSTEM_PROMPT,
    CHINESE_TO_BRAILLE_SYSTEM_PROMPT,
    ENGLISH_TO_BRAILLE_SYSTEM_PROMPT,
    BRAILLE_TO_ENGLISH_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)

# ----------------------------
# 환경 설정
# ----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

KOREAN_ENDPOINT = st.secrets["KOREAN_ENDPOINT"]
CHINESE_ENDPOINT = st.secrets["CHINESE_ENDPOINT"]
ENGLISH_ENDPOINT = st.secrets["ENGLISH_ENDPOINT"]

KOREAN_API_KEY = st.secrets["KOREAN_API_KEY"]
CHINESE_API_KEY = st.secrets["CHINESE_API_KEY"]
ENGLISH_API_KEY = st.secrets["ENGLISH_API_KEY"]

USE_SENTENCE_LEVEL_TRANSLATION = True

model_configs = {
    "qwen3-1.7b": {
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "end_token": "<|im_end|>",
    },
    "kanana-1.5-2.1b": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "end_token": "<|eot_id|>",
    },
    "kanana-1.5-2.1b-english": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "end_token": "<|eot_id|>",
    },
}


# ----------------------------
# LLM Utility
# ----------------------------
def pick_model(language: str) -> str | None:
    if language == "Korean":
        return "kanana-1.5-2.1b"
    elif language == "Chinese":
        return "qwen3-1.7b"
    elif language == "English":
        return "kanana-1.5-2.1b-english"
    else:
        return None


def pick_endpoint(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang == "korean":
        return KOREAN_ENDPOINT
    elif lang == "chinese":
        return CHINESE_ENDPOINT
    elif lang == "english":
        return ENGLISH_ENDPOINT
    return KOREAN_ENDPOINT


def pick_api_key(language: str) -> str | None:
    lang = (language or "").strip().lower()
    if lang == "korean":
        return KOREAN_API_KEY
    elif lang == "chinese":
        return CHINESE_API_KEY
    elif lang == "english":
        return ENGLISH_API_KEY
    return None


def llm_chat(system_msg: str, user_msg: str, language: str = "Korean") -> str:
    headers = {"Content-Type": "application/json"}
    api_key = pick_api_key(language).strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_name = pick_model(language)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": model_configs[model_name]["temperature"],
        "top_p": model_configs[model_name]["top_p"],
        "top_k": model_configs[model_name]["top_k"],
        "min_p": model_configs[model_name]["min_p"],
        "stop": [model_configs[model_name]["end_token"]],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if language.lower() == "chinese":
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    endpoint = pick_endpoint(language)

    # [추가됨] HTTP Request 로깅
    logger.info(f"==== [HTTP Request] LLM Chat ({model_name}) ====")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # [추가됨] HTTP Response 로깅
        logger.info("==== [HTTP Response] LLM Chat ====")
        logger.info(f"Data: {json.dumps(data, ensure_ascii=False)}")

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise ValueError("LLM returned empty content")
        return content
    except Exception as e:
        logger.error(f"==== [HTTP Error] LLM Chat ====\n{e}")
        return f"[LLM Error] {e}"


# --- Helper for normalization ---
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def gemini_summarize(text: str, source_language: str) -> str:
    normalized_input = normalize_text(text)

    progress_text = "요약 중..."
    my_bar = st.progress(0, text=progress_text)

    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = genai.types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
    )

    my_bar.progress(30, text="Gemini에 요약 요청 중...")

    # [추가됨] Gemini Request 로깅
    contents_payload = f"source_language: {source_language}\n\ninput_text: {text}"
    logger.info("==== [Gemini Request: Summarize] ====")
    logger.info(f"Contents: \n{contents_payload}")

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=cfg,
        contents=contents_payload,
    )
    my_bar.progress(90, text="요약 결과 처리 중...")

    # [추가됨] Gemini Response 로깅
    logger.info("==== [Gemini Response: Summarize] ====")
    logger.info(f"Response Text: {resp.text}")

    raw = resp.text
    parsed = json.loads(raw)
    summary = parsed.get("output_text", text)

    my_bar.progress(100, text="완료.")
    time.sleep(0.5)
    my_bar.empty()
    return summary


# ----------------------------
# Utils
# ----------------------------
_SENT_SPLIT = re.compile(
    r"""
    (?<=[.?!])\s+ | (?<=[。。！？]) | (?<=[⠲⠖⠦⠐⠆⠰⠂⠐⠄])\s+
    """,
    re.X,
)


def split_sentences_keep_punct(text: str) -> list[str]:
    if not text.strip():
        return []
    _t = text.replace("\n", " ").strip()
    parts = []
    start = 0
    for m in _SENT_SPLIT.finditer(_t):
        end = m.end()
        parts.append(_t[start:end].strip())
        start = end
    if start < len(_t):
        parts.append(_t[start:].strip())
    return [p for p in parts if p]


def sentenceize_with_line_map(text: str):
    lines = text.splitlines()
    all_sents, line_counts = [], []
    for line in lines:
        if line.strip():
            sents = split_sentences_keep_punct(line)
            if not sents:
                sents = [line.strip()]
            line_counts.append(len(sents))
            all_sents.extend(sents)
        else:
            line_counts.append(0)
    return all_sents, line_counts, lines


def assemble_by_lines(unit_list: list[str], line_counts: list[int]) -> str:
    out_lines, i = [], 0
    for cnt in line_counts:
        if cnt == 0:
            out_lines.append("")
        else:
            out_lines.append(" ".join(unit_list[i : i + cnt]))
            i += cnt
    return "\n".join(out_lines)


def _safe_str(x, fallback=""):
    return x if isinstance(x, str) and x != "" else fallback


# ----------------------------
# Translation
# ----------------------------
def run_translation(text: str) -> str:
    logger.info("==== [Translation] ====")
    logger.info(f"[Input Text]\n{text}")

    progress_text = "AI 모델 연결 중..."
    my_bar = st.progress(0, text=progress_text)

    # 2. 실제 로직
    try:
        if st.session_state.mode == "text_to_braille":
            if st.session_state.src_lang == "Korean":
                system_msg = KOREAN_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "Chinese":
                system_msg = CHINESE_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.src_lang == "English":
                system_msg = ENGLISH_TO_BRAILLE_SYSTEM_PROMPT
        else:
            if st.session_state.tgt_lang == "Korean":
                system_msg = BRAILLE_TO_KOREAN_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "Chinese":
                system_msg = BRAILLE_TO_CHINESE_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "English":
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT
            else:
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT

        lang = (
            st.session_state.src_lang
            if st.session_state.mode == "text_to_braille"
            else st.session_state.tgt_lang
        )

        if USE_SENTENCE_LEVEL_TRANSLATION:
            src_sents, line_counts, _lines = sentenceize_with_line_map(text)
            st.session_state["src_sents"] = src_sents
            st.session_state["line_counts"] = line_counts

            tgt_sents = []
            total_sents = len(src_sents)
            for i, s in enumerate(src_sents, 1):
                progress_percent = int((i / total_sents) * 100)
                my_bar.progress(
                    progress_percent, text=f"문장 번역 중 {i}/{total_sents}..."
                )
                out = llm_chat(system_msg, s, lang)
                tgt_sents.append(_safe_str(out, "[번역 없음]"))

            st.session_state["tgt_sents"] = tgt_sents
            ui_text = assemble_by_lines(tgt_sents, line_counts)

            my_bar.progress(100, text="완료.")
            time.sleep(0.5)
            my_bar.empty()
            return ui_text
        else:
            lines = text.split("\n")
            total_lines = len(lines)
            translated_lines = []
            for i, line in enumerate(lines):
                if line.strip():
                    my_bar.progress(
                        int(((i + 1) / total_lines) * 100),
                        text=f"줄 번역 중 {i+1}/{total_lines}...",
                    )
                    out = llm_chat(system_msg, line, lang)
                    translated_lines.append(_safe_str(out, "[번역 없음]"))
                else:
                    translated_lines.append("")
            result = "\n".join(translated_lines)
            my_bar.progress(100, text="완료.")
            time.sleep(0.5)
            my_bar.empty()
            return result

    except Exception as e:
        my_bar.empty()
        return f"번역 오류: {e}"


# ----------------------------
# Validation
# ----------------------------
def validate_translation(src: str, tgt_ui_text: str) -> dict:
    progress_text = "번역 검증 중..."
    val_bar = st.progress(0, text=progress_text)

    try:
        # 프롬프트 설정
        if st.session_state.mode == "text_to_braille":
            if st.session_state.src_lang == "Korean":
                system_msg = BRAILLE_TO_KOREAN_SYSTEM_PROMPT
            elif st.session_state.src_lang == "Chinese":
                system_msg = BRAILLE_TO_CHINESE_SYSTEM_PROMPT
            else:
                system_msg = BRAILLE_TO_ENGLISH_SYSTEM_PROMPT
        else:
            if st.session_state.tgt_lang == "Korean":
                system_msg = KOREAN_TO_BRAILLE_SYSTEM_PROMPT
            elif st.session_state.tgt_lang == "Chinese":
                system_msg = CHINESE_TO_BRAILLE_SYSTEM_PROMPT
            else:
                system_msg = ENGLISH_TO_BRAILLE_SYSTEM_PROMPT

        lang = (
            st.session_state.src_lang
            if st.session_state.mode == "text_to_braille"
            else st.session_state.tgt_lang
        )
        val_bar.progress(10, text="검증 준비 중...")

        validation_total_start = time.time()
        back_translation_start = time.time()
        semantic_time_total = 0.0
        logs = []

        if USE_SENTENCE_LEVEL_TRANSLATION:
            tgt_sents = st.session_state.get("tgt_sents")
            # 버퍼가 없으면 생성
            if not tgt_sents:
                tgt_sents, _, _ = sentenceize_with_line_map(tgt_ui_text)

            src_sents = st.session_state.get("src_sents")
            if not src_sents:
                src_sents, _, _ = sentenceize_with_line_map(src)

            if not src_sents:
                src_sents = [src]

            if not tgt_sents:
                tgt_sents = [tgt_ui_text]

            sentence_results = []
            recon_sents = []

            total = len(tgt_sents)

            for i in range(total):
                tgt_sent = tgt_sents[i] if i < len(tgt_sents) else ""
                src_sent = src_sents[i] if i < len(src_sents) else ""

                val_bar.progress(
                    10 + int(((i + 1) / max(total, 1)) * 40),
                    text=f"역번역 중 ({i+1}/{total})...",
                )

                if not _safe_str(tgt_sent):
                    recon_sents.append("")
                    sentence_results.append(
                        {
                            "index": i + 1,
                            "source": src_sent,
                            "braille": tgt_sent,
                            "recon": "",
                            "ok": False,
                            "method": "empty",
                            "reason": "빈 문장",
                        }
                    )
                    logs.append(f"[Sentence {i+1}] empty target")
                    continue

                # 1) 역번역
                recon = llm_chat(system_msg, tgt_sent, lang)
                recon = _safe_str(recon, "")
                recon_sents.append(recon)
                logs.append(f"[Sentence {i+1}] back-translation done")

                # 2) exact check
                if normalize_text(recon) == normalize_text(src_sent):
                    sentence_results.append(
                        {
                            "index": i + 1,
                            "source": src_sent,
                            "braille": tgt_sent,
                            "recon": recon,
                            "ok": True,
                            "method": "exact",
                            "reason": "exact match",
                        }
                    )
                    logs.append(f"[Sentence {i+1}] exact match")
                    continue

                # 3) semantic check
                semantic_start = time.time()

                client = genai.Client(api_key=GEMINI_API_KEY)
                cfg = genai.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                    system_instruction=VALIDATION_SYSTEM_PROMPT,
                )
                contents = f"src: {src_sent}\n\nrecon: {recon}"

                logger.info("==== [Gemini Request: Sentence Validation] ====")
                logger.info(f"Contents: \n{contents}")

                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=cfg,
                    contents=contents,
                )

                logger.info("==== [Gemini Response: Sentence Validation] ====")
                logger.info(f"Response: {resp.text}")

                parsed = json.loads(resp.text)
                semantic_ok = parsed.get("equal") is True

                semantic_time_total += time.time() - semantic_start

                if semantic_ok:
                    sentence_results.append(
                        {
                            "index": i + 1,
                            "source": src_sent,
                            "braille": tgt_sent,
                            "recon": recon,
                            "ok": True,
                            "method": "semantic",
                            "reason": "semantic match",
                        }
                    )
                    logs.append(f"[Sentence {i+1}] semantic match")
                else:
                    sentence_results.append(
                        {
                            "index": i + 1,
                            "source": src_sent,
                            "braille": tgt_sent,
                            "recon": recon,
                            "ok": False,
                            "method": "fail",
                            "reason": "semantic mismatch",
                        }
                    )
                    logs.append(f"[Sentence {i+1}] semantic mismatch")

            back_translation_time = time.time() - back_translation_start

            recon_joined = " ".join([s for s in recon_sents if s])

            val_bar.progress(70, text="문장별 검증 결과 집계 중...")

            all_ok = all(item["ok"] for item in sentence_results)
            fail_count = sum(1 for item in sentence_results if not item["ok"])

            recon_joined = " ".join([s for s in recon_sents if s])

            validation_total_time = time.time() - validation_total_start

            val_bar.progress(100, text="검증 완료.")
            time.sleep(0.3)
            val_bar.empty()

            if all_ok:
                msg = "✅ 전체 문장 검증 완료"
            else:
                msg = f"⚠ {fail_count}/{len(sentence_results)} 문장에서 검수 필요"

            return {
                "is_valid": all_ok,
                "message": msg,
                "recon_text": recon_joined,
                "sentence_results": sentence_results,
                "times": {
                    "back_translation": back_translation_time,
                    "semantic": semantic_time_total,
                    "total_validation": validation_total_time,
                },
                "logs": logs,
            }
        else:
            # 줄단위 검증 로직
            tgt_lines = tgt_ui_text.split("\n")
            recon_lines = []
            total = len(tgt_lines)
            for i, t_line in enumerate(tgt_lines):
                val_bar.progress(10 + int(((i + 1) / total) * 40), text="역번역 중...")
                if t_line.strip():
                    recon_lines.append(llm_chat(system_msg, t_line, lang))
                else:
                    recon_lines.append("")
            recon = "\n".join(recon_lines)

            val_bar.progress(60, text="텍스트 비교 중...")
            if recon == src:
                val_bar.empty()
                return {
                    "is_valid": True,
                    "message": "자동 검증 성공 (정방향-역방향 일치).",
                    "recon_text": "",
                    "sentence_results": [],
                    "times": {
                        "back_translation": 0,
                        "semantic": 0,
                        "total_validation": 0,
                    },
                    "logs": [],
                }

            val_bar.progress(70, text="의미 일치 여부 확인 중...")
            client = genai.Client(api_key=GEMINI_API_KEY)
            cfg = genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                system_instruction=VALIDATION_SYSTEM_PROMPT,
            )
            contents = f"src: {src}\n\nrecon: {recon}"

            # [추가됨] Gemini Request 로깅
            logger.info("==== [Gemini Request: Validation] ====")
            logger.info(f"Contents: \n{contents}")

            resp = client.models.generate_content(
                model="gemini-2.5-flash", config=cfg, contents=contents
            )

            # [추가됨] Gemini Response 로깅
            logger.info("==== [Gemini Response: Validation] ====")
            logger.info(f"Response: {resp.text}")

            parsed = json.loads(resp.text)
            val_bar.empty()
            if parsed.get("equal") is True:
                return (
                    True,
                    "자동 검증 실패 (정방향-역방향 불일치), 의미 기반 검증 성공.",
                )
            else:
                return (
                    False,
                    "자동 검증 실패 (정방향-역방향 불일치), 의미 기반 검증 실패. 추가 검토가 필요합니다.",
                )

    except Exception as e:
        val_bar.empty()
        return {
            "is_valid": False,
            "message": f"검증 오류: {e}",
            "recon_text": "",
            "sentence_results": [],
            "times": {
                "back_translation": 0,
                "semantic": 0,
                "total_validation": 0,
            },
            "logs": [str(e)],
        }


# ----------------------------
# UI Configuration & State
# ----------------------------
st.set_page_config(page_title="AI 기반 점자 번역 시스템", page_icon="⠁", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .stElementContainer { margin-bottom: -0.5rem; }
        .streamlit-expander { margin-bottom: 0px !important; }
    </style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="text-align:center; margin-top: 0rem">
      <h1 style="margin-bottom:0.2rem; font-size: 3.0rem; line-height:1.25;">
       AI 기반 점자 번역 시스템
      </h1>
    </div>
    <hr style="margin-top: 0.5rem; margin-bottom: 1.5rem; border: 0; border-top: 1px solid #f0f2f6;" />
""",
    unsafe_allow_html=True,
)

if "mode" not in st.session_state:
    st.session_state.mode = "text_to_braille"
if "src_text" not in st.session_state:
    st.session_state.src_text = ""
if "tgt_text" not in st.session_state:
    st.session_state.tgt_text = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""  # 요약문 저장
if "src_lang_val" not in st.session_state:
    st.session_state.src_lang_val = "Korean"
if "tgt_lang_val" not in st.session_state:
    st.session_state.tgt_lang_val = "Braille"
if "pending_swap" not in st.session_state:
    st.session_state.pending_swap = False
if "last_is_valid" not in st.session_state:
    st.session_state.last_is_valid = None

# [변경됨] Swap Logic - 언어는 바꾸되, 텍스트는 빈 문자열로 초기화
if st.session_state.pending_swap:
    st.session_state.src_lang_val, st.session_state.tgt_lang_val = (
        st.session_state.tgt_lang_val,
        st.session_state.src_lang_val,
    )
    # 텍스트 및 상태값 모두 초기화 (기존 서로 바꾸는 코드 대신)
    st.session_state.src_text = ""
    st.session_state.tgt_text = ""
    st.session_state.summary_text = ""

    # 캐싱된 문장이나 검증 메시지가 있다면 같이 초기화
    if "last_val_msg" in st.session_state:
        st.session_state.last_val_msg = ""
    if "tgt_sents" in st.session_state:
        st.session_state.tgt_sents = []

    st.session_state.pending_swap = False

st.session_state.src_lang = st.session_state.src_lang_val
st.session_state.tgt_lang = st.session_state.tgt_lang_val
TEXT_LANGS = ["Korean", "Braille"]

TEXT_LANGS_DISPLAY = {
    "Korean": "한국어",
    "Braille": "점자",
}


def _enforce_pair_and_mode():
    src = st.session_state.get("src_lang", "Korean")
    tgt = st.session_state.get("tgt_lang", "Braille")
    st.session_state.invalid_pair = not ((src == "Braille") ^ (tgt == "Braille"))
    if src == "Braille" and tgt != "Braille":
        st.session_state.mode = "braille_to_text"
    elif tgt == "Braille" and src != "Braille":
        st.session_state.mode = "text_to_braille"
    else:
        st.session_state.mode = st.session_state.get("mode", "text_to_braille")


def _update_action_disabled():
    _enforce_pair_and_mode()
    src_lang = st.session_state.src_lang
    tgt_lang = st.session_state.tgt_lang
    valid_pair = (src_lang == "Braille") ^ (tgt_lang == "Braille")
    llm_lang = src_lang if st.session_state.mode == "text_to_braille" else tgt_lang
    model_ok = pick_model(llm_lang) is not None
    st.session_state["action_disabled"] = (not valid_pair) or (not model_ok)
    st.session_state["valid_pair"] = valid_pair
    st.session_state["model_ok"] = model_ok
    st.session_state["llm_lang_eval"] = llm_lang


def _on_language_change():
    _update_action_disabled()


def _queue_swap():
    st.session_state.pending_swap = True


if "action_disabled" not in st.session_state:
    _update_action_disabled()
else:
    _update_action_disabled()


def clear_summary_on_change():
    """원본 텍스트 수정 시 요약 결과 초기화"""
    st.session_state.summary_text = ""


# ----------------------------
# Layout
# ----------------------------
mode = "Translation"
disabled = st.session_state.get("action_disabled", True)
valid_pair = st.session_state.get("valid_pair", False)
model_ok = st.session_state.get("model_ok", False)
is_text_to_braille = st.session_state.mode == "text_to_braille"

# Buttons
if mode == "Translation":
    if is_text_to_braille:
        col1, col2, col3 = st.columns([8, 1.2, 1])
        with col1:
            st.markdown('<h2 style="margin:0;">번역</h2>', unsafe_allow_html=True)
        with col2:
            go_summarize = st.button(
                "요약",
                type="secondary",
                use_container_width=True,
                disabled=disabled,
            )
        with col3:
            go_translate = st.button(
                "번역", type="primary", use_container_width=True, disabled=disabled
            )
    else:
        col1, col2 = st.columns([9, 1])
        with col1:
            st.markdown('<h2 style="margin:0;">번역</h2>', unsafe_allow_html=True)
        with col2:
            go_translate = st.button(
                "번역", type="primary", use_container_width=True, disabled=disabled
            )

if not valid_pair:
    st.warning("입력 또는 출력 중 하나만 점자여야 합니다.")
elif not model_ok:
    st.warning(f"{st.session_state.llm_lang_eval}에 대한 모델이 설정되지 않았습니다.")

header_cols = st.columns([5, 1, 5])
with header_cols[0]:
    st.selectbox(
        "입력 언어",
        TEXT_LANGS,
        format_func=lambda x: TEXT_LANGS_DISPLAY[x],
        key="src_lang_val",
        on_change=_on_language_change,
    )
with header_cols[1]:
    if st.button(
        "↔︎", help="Swap source/target", use_container_width=True, on_click=_queue_swap
    ):
        st.rerun()
with header_cols[2]:
    st.selectbox(
        "출력 언어",
        TEXT_LANGS,
        format_func=lambda x: TEXT_LANGS_DISPLAY[x],
        key="tgt_lang_val",
        on_change=_on_language_change,
    )

# Input
st.markdown(
    f'<h3 style="margin-top: 0.2rem; margin-bottom: 0.3rem;">입력 ({TEXT_LANGS_DISPLAY[st.session_state.src_lang]})</h3>',
    unsafe_allow_html=True,
)
st.session_state.src_text = st.text_area(
    "Source Text",
    height=110,
    label_visibility="collapsed",
    value=st.session_state.src_text,
    placeholder="번역할 텍스트를 입력하세요.",
    on_change=clear_summary_on_change,  # 원본 수정시 요약 초기화
)

# ----------------------------
# Logic Execution
# ----------------------------
src_nfc = unicodedata.normalize("NFC", st.session_state.src_text)

# 1. 요약 로직 (독립적)
if mode == "Translation" and "go_summarize" in locals() and go_summarize and src_nfc:
    st.session_state.last_val_msg = ""
    st.session_state.last_is_valid = None
    st.session_state.tgt_text = ""

    st.session_state.summary_text = gemini_summarize(src_nfc, st.session_state.src_lang)
    # Rerun to show summary immediately
    st.rerun()

# 2. 요약 결과 표시 (있을 경우)
if st.session_state.summary_text:
    st.markdown("**⬇️ 요약 결과**")
    st.info(st.session_state.summary_text)

# 3. Output Header
st.markdown(
    f'<h3 style="margin-top: 0rem; margin-bottom: 0.3rem;">출력 ({TEXT_LANGS_DISPLAY[st.session_state.tgt_lang]})</h3>',
    unsafe_allow_html=True,
)

# 4. 플레이스홀더 설정 (순차적 렌더링을 위해)
output_placeholder = st.empty()
validation_placeholder = st.empty()

output_placeholder.text_area(
    "Target Text",
    height=135,
    label_visibility="collapsed",
    placeholder="번역 결과가 여기에 표시됩니다.",
    key="tgt_text_widget",
)

if "last_val_msg" in st.session_state and st.session_state.last_val_msg:
    is_valid = st.session_state.get("last_is_valid")
    msg = st.session_state.last_val_msg
    if is_valid:
        validation_placeholder.success(msg)
    else:
        validation_placeholder.error(msg)


# ----------------------------
# 번역 + 검증 로직 (최종 완성 버전)
# ----------------------------
if mode == "Translation" and "go_translate" in locals() and go_translate and src_nfc:
    st.session_state.last_val_msg = ""
    st.session_state.last_is_valid = None

    validation_placeholder.empty()

    real_src = (
        st.session_state.summary_text if st.session_state.summary_text else src_nfc
    )

    total_start = time.time()

    # 1️⃣ 번역
    progress_text = st.empty()
    progress_text.info("1️⃣ 번역 중...")

    t_start = time.time()
    result = run_translation(real_src)
    translation_time = time.time() - t_start

    st.session_state.tgt_text = result

    output_placeholder.text_area(
        "Target Text",
        value=result,
        height=135,
        label_visibility="collapsed",
    )

    # 2️⃣ 검증 (🔥 핵심: 여기서만 수행)
    progress_text.info("2️⃣ 검증 중...")

    validation_result = validate_translation(real_src, result)

    is_valid = validation_result["is_valid"]
    val_msg = validation_result["message"]
    recon_text = validation_result["recon_text"]
    sentence_results = validation_result["sentence_results"]
    validation_times = validation_result["times"]
    logs = validation_result["logs"]

    back_time = validation_times["back_translation"]
    semantic_time = validation_times["semantic"]
    verify_time = validation_times["total_validation"]

    total_time = time.time() - total_start

    progress_text.success("✅ 처리 완료")

    # ----------------------------
    # ⏱ 시간 표시
    # ----------------------------
    st.markdown("### ⏱ 처리 시간")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("번역", f"{translation_time:.2f}s")
    col2.metric("역번역", f"{back_time:.2f}s")
    col3.metric("Semantic", f"{semantic_time:.2f}s")
    col4.metric("검증", f"{verify_time:.2f}s")
    col5.metric("총 시간", f"{total_time:.2f}s")

    # ----------------------------
    # 🔁 역번역 결과
    # ----------------------------
    st.markdown("### 🔁 역번역 결과")
    st.text_area("", value=recon_text, height=120)

    # ----------------------------
    # 🧠 문장별 검증 결과 (🔥 핵심 개선)
    # ----------------------------
    st.markdown("### 🧠 문장별 검증 결과")

    fail_count = 0

    for item in sentence_results:
        idx = item["index"]

        if item["ok"] and item["method"] == "exact":
            with st.container():
                st.markdown(f"🟢 **문장 {idx} (Exact 일치)**")
                st.write("원문:", item["source"])
                st.write("점자:", item["braille"])
                st.write("역번역:", item["recon"])

        elif item["ok"] and item["method"] == "semantic":
            with st.container():
                st.markdown(f"🟡 **문장 {idx} (Semantic 일치)**")
                st.write("원문:", item["source"])
                st.write("점자:", item["braille"])
                st.write("역번역:", item["recon"])
                st.info("Exact 불일치지만 의미는 동일")

        else:
            fail_count += 1

            with st.container():
                st.markdown(f"🔴 **문장 {idx} (검수 필요)**")
                st.error("⚠ 의미까지 불일치")
                st.write("원문:", item["source"])
                st.write("점자:", item["braille"])
                st.write("역번역:", item["recon"])

    total = len(sentence_results)

    if fail_count > 0:
        st.warning(f"⚠ {fail_count}/{total} 문장에서 검수 필요")
    else:
        st.success("✅ 모든 문장 자동 검증 완료")

    # ----------------------------
    # 🎯 최종 결과
    # ----------------------------
    st.session_state.last_val_msg = val_msg
    st.session_state.last_is_valid = is_valid

    if is_valid:
        validation_placeholder.success(val_msg)
    else:
        validation_placeholder.error(val_msg)

    # ----------------------------
    # 🔍 로그
    # ----------------------------
    st.markdown("### 🔍 처리 로그")

    with st.expander("로그 보기"):
        for log in logs[:30]:
            st.text(log)
