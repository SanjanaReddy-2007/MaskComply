"""
MaskComply -- Live Compliance Dashboard (v2)
=============================================
Runs the detect -> track -> score pipeline on a video and shows, live:

  * the annotated video (boxes only / boxes + IDs / full labels, optional score banner)
  * a compliance gauge with the Green / Yellow / Red band
  * KPI cards (session score, people tracked, mask share, alerts, speed)
  * a trend chart with the band zones, the RAW per-frame score next to the
    EMA-SMOOTHED score (this is the smoothing ablation, live) and alert markers
  * a class-composition chart, a "lowest-scoring people right now" table
  * an alert feed (band crossings only) and a session summary with CSV / PNG export

Usage (from the repo folder):
    streamlit run src/dashboard.py
    streamlit run src/dashboard.py -- --video <clip> --yolo_weights models/best.pt

With --video the analysis starts immediately (same as before); without it, pick
a clip in the sidebar and press "Start analysis". Pressing "Stop" freezes the
current frame so you can download a clean snapshot for the report.
"""

import argparse
import csv
import io
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:  # altair ships with streamlit, so this should not happen
    alt = None

try:
    from detect import CLASS_COLORS
except Exception:  # detect.py pulls in torch; colours are all we need from it
    CLASS_COLORS = {"mask": (0, 200, 0), "no_mask": (0, 0, 220), "incorrect": (0, 200, 220)}

from detection import add_detector_args, make_detector, classify_tracks
from tracker import CentroidTracker
from scorer import ComplianceAggregator, raw_score

REPO = Path(__file__).resolve().parent.parent

BAND_HEX = {"GREEN": "#22c55e", "YELLOW": "#f59e0b", "RED": "#ef4444"}
BAND_BGR = {"GREEN": (94, 197, 34), "YELLOW": (11, 158, 245), "RED": (68, 68, 239)}
BAND_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}
CLASS_HEX = {"mask": "#22c55e", "incorrect": "#f59e0b", "no_mask": "#ef4444"}
CLASS_NICE = {"mask": "Mask", "incorrect": "Incorrect", "no_mask": "No mask"}

OVERLAYS = ["Boxes only", "Boxes + ID", "Full labels"]


# --------------------------------------------------------------------------- #
# Arguments / paths
# --------------------------------------------------------------------------- #
def parse_args():
    # Streamlit hands the script everything after the script name, but strips the
    # "--" separator you type on the command line. Older versions kept it, so
    # accept both.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="clip to analyse (path, or a name from data/test_clips); starts immediately")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max_missed", type=int, default=10)
    parser.add_argument("--alert_log", default="outputs/alert_log.csv")
    parser.add_argument("--empty_score", type=float, default=1.0,
                        help="score shown when nobody is detected (default 1.0 = green; -1 = none)")
    add_detector_args(parser)
    args, _unknown = parser.parse_known_args(argv)
    return args


def resolve_path(value, extra_dirs=()):
    """Find a file given as an absolute path, a cwd-relative path, or relative to the repo."""
    if not value:
        return None
    p = Path(value)
    cands = [p, REPO / p] + [d / p for d in extra_dirs]
    if p.suffix == "":
        cands += [d / (str(value) + ".mp4") for d in extra_dirs]
    for c in cands:
        if c.exists() and c.is_file():
            return c
    return None


def list_clips():
    clips = {}
    for root in (Path.cwd() / "data" / "test_clips", REPO / "data" / "test_clips"):
        if root.is_dir():
            for f in sorted(root.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".m4v"):
                    clips.setdefault(f.name, f)
    return clips


def fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container{padding-top:1.3rem;padding-bottom:2rem;max-width:1600px}
footer{visibility:hidden}
.mc-hero{display:flex;align-items:center;gap:14px}
.mc-logo{font-size:2.1rem;line-height:1}
.mc-title{font-size:1.65rem;font-weight:760;letter-spacing:-.02em;margin:0;line-height:1.15}
.mc-sub{color:#8b9bb8;font-size:.9rem;margin-top:2px}
.mc-pill{display:inline-block;padding:5px 14px;border-radius:999px;font-size:.78rem;font-weight:700;
  letter-spacing:.08em;border:1px solid #1f2a44;background:#111a2e;color:#8b9bb8}
.mc-pill.live{color:#22c55e;border-color:#14532d;background:#0c2417}
.mc-pill.done{color:#38bdf8;border-color:#0c4a6e;background:#0a2233}
.mc-pill.stop{color:#f59e0b;border-color:#78350f;background:#2a1c08}
.mc-card{background:linear-gradient(180deg,#131d33,#0f1729);border:1px solid #1f2a44;border-radius:14px;
  padding:13px 16px;min-height:98px}
.mc-label{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:#8b9bb8;font-weight:650}
.mc-value{font-size:1.85rem;font-weight:760;line-height:1.18;margin-top:2px}
.mc-note{color:#8b9bb8;font-size:.78rem;margin-top:2px}
.mc-gauge{display:flex;align-items:center;gap:18px}
.mc-ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;flex:none}
.mc-ring-in{width:118px;height:118px;border-radius:50%;background:#0f1729;display:flex;flex-direction:column;
  align-items:center;justify-content:center}
.mc-ring-val{font-size:2.2rem;font-weight:780;line-height:1}
.mc-ring-band{font-size:.72rem;letter-spacing:.12em;font-weight:700;margin-top:5px}
.mc-legend{display:flex;flex-direction:column;gap:6px;font-size:.82rem;color:#b6c3dc}
.mc-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
.mc-feed{max-height:250px;overflow:auto}
.mc-alert{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #1b2742;font-size:.84rem}
.mc-alert:last-child{border-bottom:none}
.mc-chip{padding:2px 9px;border-radius:6px;font-size:.7rem;font-weight:700;letter-spacing:.05em;color:#0b1220}
.mc-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:6px 0 14px 0}
.mc-step{flex:1;min-width:150px;background:#111a2e;border:1px solid #1f2a44;border-radius:12px;padding:11px 13px}
.mc-step b{display:block;font-size:.9rem;margin-bottom:3px}
.mc-step span{color:#8b9bb8;font-size:.78rem}
.mc-arrow{align-self:center;color:#38bdf8;font-size:1.2rem}
.mc-bar{display:flex;height:26px;border-radius:8px;overflow:hidden;border:1px solid #1f2a44}
.mc-bar div{display:flex;align-items:center;justify-content:center;font-size:.74rem;font-weight:700;color:#0b1220}
.mc-empty{border:1px dashed #2a3a5e;border-radius:16px;padding:34px 30px;background:#0f1729}
</style>
"""


def card(label, value, note="", color=None):
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="mc-card"><div class="mc-label">{label}</div>'
            f'<div class="mc-value"{style}>{value}</div><div class="mc-note">{note}</div></div>')


def gauge_html(S):
    rows = S["rows"] if S else []
    if not rows or rows[-1]["frame_score"] is None:
        value, band, pct, color, note = "&ndash;", "NO SCORE", 0, "#475569", "no people in frame"
        if rows:
            note = "no people in frame (empty-scene score disabled)"
        else:
            note = "waiting for the first frame"
    else:
        r = rows[-1]
        value, band = f"{r['frame_score']:.2f}", r["band"]
        pct = max(0.0, min(1.0, r["frame_score"])) * 100
        color = BAND_HEX[band]
        note = "no people detected &mdash; scene is empty" if r["no_people"] else \
            f"{r['n_people']} tracked &middot; mean of per-person EMA scores"
    ring_bg = f"conic-gradient({color} 0 {pct:.1f}%, #1f2a44 {pct:.1f}% 100%)"
    legend = "".join(
        f'<div><span class="mc-dot" style="background:{BAND_HEX[b]}"></span>{b.title()} &nbsp;<span style="color:#8b9bb8">{rng}</span></div>'
        for b, rng in (("GREEN", "&ge; 0.80"), ("YELLOW", "0.50 &ndash; 0.80"), ("RED", "&lt; 0.50")))
    return (
        '<div class="mc-card" style="min-height:0"><div class="mc-label">Current frame score</div>'
        '<div class="mc-gauge" style="margin-top:10px">'
        f'<div class="mc-ring" style="background:{ring_bg}"><div class="mc-ring-in">'
        f'<div class="mc-ring-val" style="color:{color}">{value}</div>'
        f'<div class="mc-ring-band" style="color:{color}">{band}</div></div></div>'
        f'<div class="mc-legend">{legend}</div></div>'
        f'<div class="mc-note" style="margin-top:10px">{note}</div></div>')


def alert_feed_html(S, limit=6):
    alerts = S["alerts"] if S else []
    if not alerts:
        body = '<div class="mc-note" style="padding:6px 0">No band changes yet.</div>'
    else:
        items = []
        for a in reversed(alerts[-limit:]):
            up = BAND_RANK[a["to"]] > BAND_RANK[a["from"]]
            arrow = "&#9650;" if up else "&#9660;"
            kind = "Escalation" if up else "Recovery"
            kcolor = "#ef4444" if up else "#22c55e"
            items.append(
                f'<div class="mc-alert"><span style="color:{kcolor};font-size:.8rem">{arrow}</span>'
                f'<span class="mc-chip" style="background:{BAND_HEX[a["from"]]}">{a["from"]}</span>&rarr;'
                f'<span class="mc-chip" style="background:{BAND_HEX[a["to"]]}">{a["to"]}</span>'
                f'<span style="flex:1;color:#b6c3dc">{kind} &middot; score {a["score"]:.2f}</span>'
                f'<span style="color:#8b9bb8">{fmt_time(a["t"])} &middot; f{a["frame"]}</span></div>')
        body = "".join(items)
    n = len(alerts)
    return (f'<div class="mc-card" style="min-height:0"><div class="mc-label">Alert feed '
            f'&middot; {n} band change{"s" if n != 1 else ""}</div>'
            f'<div class="mc-feed" style="margin-top:6px">{body}</div></div>')


def status_pill(status):
    cls, text = {"running": ("live", "&#9679; LIVE"), "finished": ("done", "COMPLETE"),
                 "stopped": ("stop", "STOPPED"), "idle": ("", "READY")}[status]
    return f'<span class="mc-pill {cls}">{text}</span>'


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def stretch(fn, *args, **kwargs):
    """Call a Streamlit element with 'full width' on both old and new Streamlit versions."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*args, **extra, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # StreamlitAPIException for an unsupported width value
            if "width" in str(exc).lower():
                continue
            raise


def new_run(meta):
    return {
        "meta": meta, "status": "running",
        "rows": [], "alerts": [],
        "labels": {}, "age": {}, "cur": {"ema": {}, "det": {}, "missed": set()},
        "sum": 0.0, "cnt": 0, "bands": {"GREEN": 0, "YELLOW": 0, "RED": 0},
        "empty": 0, "peak": 0, "unique": 0,
        "jr": 0.0, "je": 0.0, "jn": 0, "prev_raw": None, "prev_ema": None,
        "last_img": None, "last_band": None,
        "processed": 0, "elapsed": 0.0, "frame_idx": 0,
    }


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def annotate(frame, active_boxes, per_person, tracker, mode, banner, max_w=1100, max_h=720):
    """Returns an RGB image scaled for display with the overlay drawn on it."""
    h, w = frame.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    img = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) \
        if s < 1.0 else frame.copy()
    ih, iw = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tid, (x, y, bw, bh) in active_boxes.items():
        if not tracker.is_confirmed(tid):
            continue
        x0, y0 = int(round(x * s)), int(round(y * s))
        x1, y1 = int(round((x + bw) * s)), int(round((y + bh) * s))
        if tid in per_person:
            label, conf = per_person[tid]
            color = CLASS_COLORS.get(label, (255, 255, 255))
            text = {"Boxes only": "", "Boxes + ID": f"#{tid}",
                    "Full labels": f"#{tid} {label} {conf:.2f}"}[mode]
            thick = 2
        else:
            color, text, thick = (150, 150, 150), ("" if mode == "Boxes only" else f"#{tid} (held)"), 1
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
        if text:
            (tw, th), _ = cv2.getTextSize(text, font, 0.42, 1)
            ty0 = y0 - th - 6
            if ty0 < 0:
                ty0 = y0
            cv2.rectangle(img, (x0, ty0), (x0 + tw + 6, ty0 + th + 6), color, -1)
            lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]   # dark text on bright chips
            ink = (20, 20, 20) if lum > 150 else (255, 255, 255)
            cv2.putText(img, text, (x0 + 3, ty0 + th + 1), font, 0.42, ink, 1, cv2.LINE_AA)

    if banner is not None:
        score, band, people = banner
        txt = f"{score:.2f}  {band}" if score is not None else "NO SCORE"
        sub = f"{people} people" if people else "no people"
        (tw, th), _ = cv2.getTextSize(txt, font, 0.8, 2)
        (sw, sh), _ = cv2.getTextSize(sub, font, 0.5, 1)
        bw_, bh_ = max(tw, sw) + 24, th + sh + 30
        x0, y0 = 10, 10
        x1, y1 = min(iw - 1, x0 + bw_), min(ih - 1, y0 + bh_)
        roi = img[y0:y1, x0:x1]
        dark = np.full_like(roi, (20, 14, 11))
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, dark, 0.65, 0)
        color = BAND_BGR.get(band, (200, 200, 200))
        cv2.rectangle(img, (x0, y0), (x0 + 5, y1), color, -1)
        cv2.putText(img, txt, (x0 + 14, y0 + th + 8), font, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(img, sub, (x0 + 14, y0 + th + sh + 20), font, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def frames_df(S):
    df = pd.DataFrame(S["rows"])
    step = max(1, len(df) // 700)
    if step > 1:
        df = pd.concat([df.iloc[::step], df.iloc[[-1]]]).drop_duplicates("frame")
    return df


def trend_chart(S):
    df = frames_df(S)
    ema = df["frame_score"].where(~df["no_people"])
    plot = pd.DataFrame({
        "t": df["t"],
        "Raw detections": df["raw"],
        "Smoothed (EMA)": ema,
        "Session average": df["session"],
    })
    series = ["Raw detections", "Smoothed (EMA)", "Session average"]
    colors = ["#64748b", "#38bdf8", "#e2e8f0"]
    dur = S["meta"].get("duration")
    x_scale = alt.Scale(domain=[0, dur]) if dur else alt.Scale(zero=True)

    bands = pd.DataFrame({"lo": [0.0, 0.5, 0.8], "hi": [0.5, 0.8, 1.0],
                          "color": [BAND_HEX["RED"], BAND_HEX["YELLOW"], BAND_HEX["GREEN"]]})
    band_layer = alt.Chart(bands).mark_rect(opacity=0.10).encode(
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1]), title="Compliance score"),
        y2="hi:Q", color=alt.Color("color:N", scale=None, legend=None))

    lines = alt.Chart(plot).transform_fold(series, as_=["Series", "Score"]).mark_line().encode(
        x=alt.X("t:Q", title="Video time (s)", scale=x_scale),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=colors),
                        legend=alt.Legend(orient="top", title=None)),
        size=alt.Size("Series:N", scale=alt.Scale(domain=series, range=[1.2, 3.0, 1.8]), legend=None),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
            domain=series, range=[[1, 0], [1, 0], [6, 4]]), legend=None),
    )

    layers = [band_layer, lines]
    al = [a for a in S["alerts"]][-30:]
    if al:
        rules = pd.DataFrame({"t": [a["t"] for a in al], "color": [BAND_HEX[a["to"]] for a in al]})
        layers.append(alt.Chart(rules).mark_rule(opacity=0.55, strokeDash=[3, 3]).encode(
            x=alt.X("t:Q", scale=x_scale), color=alt.Color("color:N", scale=None, legend=None)))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=330).configure_axis(
        gridOpacity=0.15).configure_view(strokeWidth=0)


def composition_chart(S):
    df = frames_df(S)
    plot = pd.DataFrame({"t": df["t"], "Mask": df["mask"], "Incorrect": df["incorrect"],
                         "No mask": df["no_mask"]})
    dur = S["meta"].get("duration")
    x_scale = alt.Scale(domain=[0, dur]) if dur else alt.Scale(zero=True)
    names = ["Mask", "Incorrect", "No mask"]
    return alt.Chart(plot).transform_fold(names, as_=["Class", "People"]).transform_calculate(
        rank="indexof(['Mask','Incorrect','No mask'], datum.Class)").mark_area(
        opacity=0.85, interpolate="step-after").encode(
        x=alt.X("t:Q", title="Video time (s)", scale=x_scale),
        y=alt.Y("People:Q", stack="zero", title="People tracked"),
        color=alt.Color("Class:N", scale=alt.Scale(
            domain=names, range=[CLASS_HEX["mask"], CLASS_HEX["incorrect"], CLASS_HEX["no_mask"]]),
            legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("rank:Q", sort="ascending"),
    ).properties(height=330).configure_axis(gridOpacity=0.15).configure_view(strokeWidth=0)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def jitter_reduction(S):
    if S["jn"] >= 10 and S["jr"] > 1e-9:
        return max(0.0, 1.0 - S["je"] / S["jr"])
    return None


def render_kpis(phs, S):
    meta = S["meta"]
    rows = S["rows"]
    last = rows[-1] if rows else None

    if S["cnt"]:
        avg = S["sum"] / S["cnt"]
        phs[0].markdown(card("Session score", f"{avg:.2f}",
                             f"mean over {S['cnt']} frames with people",
                             BAND_HEX[_band(avg)]), unsafe_allow_html=True)
    else:
        phs[0].markdown(card("Session score", "&ndash;", "no frames with people yet"),
                        unsafe_allow_html=True)

    n = last["n_people"] if last else 0
    phs[1].markdown(card("People tracked", f"{n}",
                         f"peak {S['peak']} &middot; {S['unique']} unique IDs"), unsafe_allow_html=True)

    if last and last["n_people"]:
        share = 100.0 * last["mask"] / last["n_people"]
        note = f"{last['mask']} mask &middot; {last['incorrect']} incorrect &middot; {last['no_mask']} no mask"
        phs[2].markdown(card("Masked right now", f"{share:.0f}%", note,
                             BAND_HEX[_band(share / 100)]), unsafe_allow_html=True)
    else:
        phs[2].markdown(card("Masked right now", "&ndash;", "nobody tracked in this frame"),
                        unsafe_allow_html=True)

    na = len(S["alerts"])
    if na:
        a = S["alerts"][-1]
        note = f"last: {a['from']} &rarr; {a['to']} at {fmt_time(a['t'])}"
    else:
        note = "no band changes yet"
    phs[3].markdown(card("Alerts", f"{na}", note), unsafe_allow_html=True)

    fps = S["processed"] / S["elapsed"] if S["elapsed"] > 0 else 0.0
    tot = meta.get("total_frames") or 0
    done = f"frame {S['frame_idx']}" + (f" / {tot}" if tot else "")
    phs[4].markdown(card("Processing speed", f"{fps:.1f} <span style='font-size:1rem'>fps</span>",
                         f"{done} &middot; source {meta['fps']:.0f} fps"), unsafe_allow_html=True)


def _band(score):
    return "GREEN" if score >= 0.8 else "YELLOW" if score >= 0.5 else "RED"


def render_people(ph, S):
    cur = S["cur"]
    with ph.container():
        if not cur["ema"]:
            st.info("Nobody is being tracked in the current frame.")
            return
        recs = []
        for tid, ema in cur["ema"].items():
            label, conf = S["labels"].get(tid, ("no_mask", 0.0))
            recs.append({"ID": f"#{tid}", "Class": CLASS_NICE.get(label, label),
                         "Confidence": round(conf, 2), "Score (EMA)": round(float(ema), 3),
                         "Status": "held (not seen this frame)" if tid in cur["missed"] else "detected",
                         "Frames tracked": S["age"].get(tid, 0)})
        df = pd.DataFrame(recs).sort_values("Score (EMA)").head(15)
        st.caption(f"Lowest-scoring people in the current frame ({len(cur['ema'])} tracked in total). "
                   f"{S['unique']} unique IDs have been confirmed so far - a very large number in a "
                   "short clip suggests ID churn from the simple centroid tracker.")
        cfg = {"Score (EMA)": st.column_config.ProgressColumn(
            "Score (EMA)", min_value=0.0, max_value=1.0, format="%.2f")}
        stretch(st.dataframe, df, hide_index=True, column_config=cfg)


def render_summary(ph, S, static):
    meta = S["meta"]
    with ph.container():
        if not S["rows"]:
            st.info("The summary appears once the first frames have been processed.")
            return
        cnt = S["cnt"]
        avg = S["sum"] / cnt if cnt else None
        c = st.columns(4)
        c[0].metric("Session score", f"{avg:.2f}" if avg is not None else "-")
        c[1].metric("Frames analysed", f"{S['processed']}")
        c[2].metric("Frames with people", f"{cnt}", f"{S['empty']} empty", delta_color="off")
        c[3].metric("Band changes", f"{len(S['alerts'])}")
        c = st.columns(4)
        c[0].metric("Peak people in frame", f"{S['peak']}")
        c[1].metric("Unique track IDs", f"{S['unique']}")
        scores = [r["frame_score"] for r in S["rows"] if not r["no_people"] and r["frame_score"] is not None]
        c[2].metric("Lowest frame score", f"{min(scores):.2f}" if scores else "-")
        jr = jitter_reduction(S)
        c[3].metric("Jitter removed by EMA", f"{100 * jr:.0f}%" if jr is not None else "-",
                    help="1 - (mean frame-to-frame change of the smoothed score / that of the raw score)")

        if cnt:
            st.markdown("**Time in each band** (frames with people)")
            parts = []
            for b in ("GREEN", "YELLOW", "RED"):
                pct = 100.0 * S["bands"][b] / cnt
                if pct > 0:
                    label = f"{pct:.0f}%" if pct >= 7 else ""
                    parts.append(f'<div style="width:{pct:.2f}%;background:{BAND_HEX[b]}">{label}</div>')
            st.markdown(f'<div class="mc-bar">{"".join(parts)}</div>', unsafe_allow_html=True)
            band = _band(avg)
            nm = 100.0 * sum(r["no_mask"] for r in S["rows"]) / max(1, sum(r["n_people"] for r in S["rows"]))
            verdict = {"GREEN": "Overall compliance is high.",
                       "YELLOW": "Compliance is mixed - a meaningful share of people are unmasked or incorrectly masked.",
                       "RED": "Compliance is low - most tracked people are not wearing a mask correctly."}[band]
            st.markdown(f"<div class='mc-note' style='margin-top:8px'>{verdict} "
                        f"({nm:.0f}% of person-frames were classed no_mask.)</div>", unsafe_allow_html=True)

        if static:
            st.markdown("**Export**")
            d = st.columns(3)
            df = pd.DataFrame(S["rows"])
            stem = Path(meta["name"]).stem
            _download(d[0], "Per-frame scores (CSV)", df.to_csv(index=False).encode(),
                      f"{stem}_frames.csv", "text/csv", "dl_frames")
            adf = pd.DataFrame(S["alerts"]) if S["alerts"] else pd.DataFrame(
                columns=["frame", "t", "from", "to", "score"])
            _download(d[1], "Alerts (CSV)", adf.to_csv(index=False).encode(),
                      f"{stem}_alerts.csv", "text/csv", "dl_alerts")
            if S["last_img"] is not None:
                ok, buf = cv2.imencode(".png", cv2.cvtColor(S["last_img"], cv2.COLOR_RGB2BGR))
                if ok:
                    _download(d[2], "Current frame (PNG)", buf.tobytes(),
                              f"{stem}_frame{S['frame_idx']}.png", "image/png", "dl_png")


def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


def _download(col, label, data, name, mime, key):
    if _st_version() >= (1, 43):   # on_click="ignore" = download without re-running the script
        col.download_button(label, data, name, mime, key=key, on_click="ignore")
    else:
        col.download_button(label, data, name, mime, key=key)


def render_about(S_unused=None):
    st.markdown(
        '<div class="mc-flow">'
        '<div class="mc-step"><b>1 &middot; Video</b><span>frames from a file or stream</span></div><div class="mc-arrow">&rarr;</div>'
        '<div class="mc-step"><b>2 &middot; YOLOv8n detector</b><span>face + class (mask / incorrect / no_mask) + confidence</span></div><div class="mc-arrow">&rarr;</div>'
        '<div class="mc-step"><b>3 &middot; Centroid tracker</b><span>persistent IDs; counts after 3 matches; holds 10 missed frames</span></div><div class="mc-arrow">&rarr;</div>'
        '<div class="mc-step"><b>4 &middot; EMA scorer</b><span>per-person smoothing, then frame and session scores</span></div><div class="mc-arrow">&rarr;</div>'
        '<div class="mc-step"><b>5 &middot; Dashboard</b><span>bands, trends, alert log</span></div></div>',
        unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Scoring**")
        st.code("class weight : mask 1.0 | incorrect 0.5 | no_mask 0.0\n"
                "raw score    = class weight x detection confidence\n"
                "EMA(p, t)    = alpha x raw + (1 - alpha) x EMA(p, t-1)\n"
                "frame score  = mean EMA of the people tracked in the frame\n"
                "bands        : Green >= 0.8 | Yellow 0.5-0.8 | Red < 0.5\n"
                "empty scene  : GREEN, not added to the session average",
                language="text")
        st.markdown("**Known limits** - small background faces can be missed (so scores may read slightly "
                    "high), no_mask recall is 0.79, the model was trained on close-up photos (domain gap "
                    "to street footage), and the centroid tracker can churn IDs in dense crowds.")
    with b:
        st.markdown("**Detector - held-out validation (171 images)**")
        st.dataframe(pd.DataFrame({
            "Class": ["All", "mask", "no_mask", "incorrect"],
            "Precision": [0.935, None, None, None],
            "Recall": [0.829, 0.956, 0.793, 0.737],
            "mAP50": [0.860, 0.982, 0.879, 0.720],
            "mAP50-95": [0.594, None, None, None]}), hide_index=True)
        st.caption("YOLOv8n, 50 epochs at 960 px, Kaggle face-mask dataset. The 'incorrect' class has only "
                   "19 validation examples, so its figures are rough.")


def render_fast(ph, S, tabs_ph=None):
    ph["status"].markdown(status_pill(S["status"]), unsafe_allow_html=True)
    ph["gauge"].markdown(gauge_html(S), unsafe_allow_html=True)
    render_kpis(ph["kpi"], S)
    meta = S["meta"]
    tot = meta.get("total_frames") or 0
    frac = min(1.0, S["frame_idx"] / tot) if tot else 0.0
    ph["progress"].progress(frac, text=f"{fmt_time(S['frame_idx'] / meta['fps'])} / "
                                       f"{fmt_time(meta.get('duration') or 0)}  -  {meta['name']}")


def render_slow(ph, S, static=False):
    ph["feed"].markdown(alert_feed_html(S), unsafe_allow_html=True)
    if S["rows"]:
        stretch(ph["trend"].altair_chart, trend_chart(S))
        stretch(ph["comp"].altair_chart, composition_chart(S))
    render_people(ph["people"], S)
    render_summary(ph["summary"], S, static)


def show_video(ph, S):
    if S["last_img"] is not None:
        ph["video"].image(S["last_img"], width=S["last_img"].shape[1], output_format="JPEG")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detector ...")
def load_detector(kind, weights, classifier_weights, conf, imgsz):
    return make_detector(kind, weights, classifier_weights, conf=conf, imgsz=imgsz)


def main():
    args = parse_args()
    st.set_page_config(page_title="MaskComply Dashboard", page_icon="😷", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    S = st.session_state.get("run")
    if S and S["status"] == "running":      # the previous run was interrupted (Stop / new click)
        S["status"] = "stopped"

    clips = list_clips()
    cli_video = resolve_path(args.video, [REPO / "data" / "test_clips"]) if args.video else None
    if args.video and cli_video is None:
        st.warning(f"Video `{args.video}` was not found - pick a clip in the sidebar instead.")
    if cli_video is not None and cli_video.name not in clips:
        clips = {cli_video.name: cli_video, **clips}
    names = list(clips.keys())
    default_idx = names.index(cli_video.name) if cli_video is not None else 0

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.markdown("### 😷 MaskComply")
        st.caption("Temporal face-mask compliance scoring")
        try:
            form = st.form("controls", border=False)
        except TypeError:  # older Streamlit without the border option
            form = st.form("controls")
        with form:
            st.markdown("**Source**")
            clip = st.selectbox("Test clip", names, index=default_idx) if names else None
            if not names:
                st.caption("No clips found in data/test_clips - upload one below.")
            upload = st.file_uploader("...or upload a video", type=["mp4", "avi", "mov", "mkv", "m4v"])
            st.markdown("**Detector**")
            conf = st.slider("Detection confidence", 0.10, 0.90, float(args.conf), 0.05,
                             help="Boxes below this confidence are dropped. Use one value for all clips.")
            sizes = [320, 480, 640, 800, 960, 1280]
            imgsz = st.select_slider("Inference size (px)", sizes,
                                     value=min(sizes, key=lambda s: abs(s - args.imgsz)),
                                     help="Smaller is faster on CPU; 960 matches training.")
            st.markdown("**Scoring**")
            alpha = st.slider("EMA smoothing (alpha)", 0.05, 1.00, float(args.alpha), 0.05,
                              help="Lower = smoother score, slower to react. 1.0 = no smoothing.")
            max_missed = st.slider("Hold a lost track (frames)", 0, 30, int(args.max_missed))
            empty_on = st.checkbox("Empty scene counts as GREEN", value=args.empty_score >= 0)
            st.markdown("**Display**")
            overlay = st.radio("Overlay", OVERLAYS, index=1, horizontal=False)
            banner_on = st.checkbox("Score banner on video", value=True)
            stride = st.slider("Process every Nth frame", 1, 10, 1,
                               help="Skips frames to speed things up on a CPU (EMA then updates per processed frame).")
            realtime = st.checkbox("Play at video speed", value=True,
                                   help="Only slows things down when the machine is faster than the video.")
            go = stretch(st.form_submit_button, "▶  Start analysis", type="primary")
        st.caption(f"Detector: {args.detector} · weights: {Path(args.yolo_weights).name if args.detector == 'yolo' else args.classifier_weights}")

    autostart = cli_video is not None and not st.session_state.get("_auto_done")
    will_run = bool(go or autostart)
    if autostart:
        st.session_state["_auto_done"] = True

    # ---------------- header ----------------
    h1, h2, h3 = st.columns([6, 1.2, 0.9])
    with h1:
        st.markdown('<div class="mc-hero"><div class="mc-logo">😷</div><div>'
                    '<div class="mc-title">MaskComply &middot; Live Compliance Dashboard</div>'
                    '<div class="mc-sub">Lightweight detection + tracking + temporal smoothing &rarr; '
                    'an interpretable 0&ndash;1 compliance score</div></div></div>', unsafe_allow_html=True)
    ph = {"status": h2.empty()}
    with h3:
        if will_run:
            st.button("⏹  Stop", help="Freeze the current frame (you can then download it)")

    kcols = st.columns(5)
    ph["kpi"] = [c.empty() for c in kcols]

    left, right = st.columns([5, 3], gap="medium")
    with left:
        ph["video"] = st.empty()
        ph["progress"] = st.empty()
    with right:
        ph["gauge"] = st.empty()
        ph["feed"] = st.empty()

    t_trend, t_comp, t_people, t_sum, t_about = st.tabs(
        ["📈  Trend", "🧩  Composition", "👥  People", "📋  Session summary", "ℹ️  How it works"])
    with t_trend:
        ph["trend"] = st.empty()
        st.caption("Grey = raw per-frame detections, blue = smoothed EMA score, dashed = session average. "
                   "Dotted vertical lines mark band changes. Gaps = no people in the frame.")
    with t_comp:
        ph["comp"] = st.empty()
    with t_people:
        ph["people"] = st.empty()
    with t_sum:
        ph["summary"] = st.empty()
    with t_about:
        render_about()

    # ---------------- idle: show the last run (or a welcome card) ----------------
    if not will_run:
        if S and S["rows"]:
            render_fast(ph, S)
            show_video(ph, S)
            render_slow(ph, S, static=True)
        else:
            idle = new_run({"name": "-", "fps": 25.0, "total_frames": 0, "duration": 0})
            idle["status"] = "idle"
            render_fast(ph, idle)
            ph["video"].markdown(
                '<div class="mc-empty"><div class="mc-title" style="font-size:1.25rem">Ready when you are</div>'
                '<div class="mc-sub" style="margin-top:8px">Pick a clip (or upload one) in the sidebar and press '
                '<b>Start analysis</b>.<br>The video, gauge, trend, composition and alert feed update live; '
                'press <b>Stop</b> to freeze a frame and download it for your report.</div></div>',
                unsafe_allow_html=True)
            ph["feed"].markdown(alert_feed_html(None), unsafe_allow_html=True)
        return

    # ---------------- resolve inputs ----------------
    tmp_dir = None
    if upload is not None:
        tmp_dir = tempfile.mkdtemp(prefix="maskcomply_")
        video_path = Path(tmp_dir) / upload.name
        video_path.write_bytes(upload.getbuffer())
        video_name = upload.name
    elif clip and clip in clips:
        video_path, video_name = clips[clip], clip
    else:
        st.error("No video selected. Put clips in data/test_clips or upload one in the sidebar.")
        return

    kind = args.detector
    weights = args.yolo_weights
    if kind == "yolo":
        w = resolve_path(weights)
        if w is None:
            st.error(f"YOLO weights not found at `{weights}`. Copy your trained `best.pt` into the "
                     "`models` folder, or start the dashboard with `-- --yolo_weights <path>`.")
            return
        weights = str(w)
    try:
        detector = load_detector(kind, weights, args.classifier_weights, float(conf), int(imgsz))
    except Exception as exc:
        st.error(f"Could not load the detector: {exc}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = total if total > 0 else 0

    empty_val = None if not empty_on else (args.empty_score if args.empty_score >= 0 else 1.0)
    meta = {"name": video_name, "fps": float(fps), "total_frames": total,
            "duration": (total / fps) if total else 0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}

    tracker = CentroidTracker(max_missed=int(max_missed), max_distance=100, min_hits=3)
    agg = ComplianceAggregator(alpha=float(alpha), session_window=10 ** 9, empty_frame_score=empty_val)

    S = new_run(meta)
    st.session_state["run"] = S

    alert_log = Path(args.alert_log)
    alert_log.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_log, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "frame", "from_band", "to_band", "score"])

    render_fast(ph, S)
    last_slow = 0.0
    t_run = time.perf_counter()
    idx = 0

    # ---------------- main loop ----------------
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector(frame)
        active = tracker.update([d.box for d in detections])
        per_person, missed = classify_tracks(tracker, active, detections)
        result = agg.update_frame(per_person, missed)

        score, band, no_people = result["frame_score"], result["frame_band"], result["no_people"]
        ema_map = result["per_person_ema"]

        for tid, (label, c) in per_person.items():
            S["labels"][tid] = (label, c)
        for tid in ema_map:
            S["age"][tid] = S["age"].get(tid, 0) + 1
        S["cur"] = {"ema": dict(ema_map), "det": dict(per_person), "missed": set(missed)}
        S["unique"] = len(agg.person_scorers)

        counts = {"mask": 0, "incorrect": 0, "no_mask": 0}
        for tid in ema_map:
            lab = S["labels"].get(tid, ("no_mask", 0.0))[0]
            counts[lab if lab in counts else "no_mask"] += 1
        n_people = len(ema_map)
        S["peak"] = max(S["peak"], n_people)

        raws = [raw_score(l, c) for (l, c) in per_person.values()]
        raw_mean = float(np.mean(raws)) if raws else None

        if no_people:
            S["empty"] += 1
            S["prev_raw"] = S["prev_ema"] = None
        else:
            S["sum"] += score
            S["cnt"] += 1
            S["bands"][band] += 1
            if raw_mean is not None and S["prev_raw"] is not None:
                S["jr"] += abs(raw_mean - S["prev_raw"])
                S["je"] += abs(score - S["prev_ema"])
                S["jn"] += 1
            S["prev_raw"] = raw_mean
            S["prev_ema"] = score

        # alert on band CROSSINGS only (same behaviour as the original dashboard)
        if band is not None and band != S["last_band"]:
            if S["last_band"] is not None:
                a = {"frame": idx, "t": idx / fps, "from": S["last_band"], "to": band, "score": score}
                S["alerts"].append(a)
                with open(alert_log, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"),
                                            idx, a["from"], a["to"], round(score, 4)])
            S["last_band"] = band

        S["rows"].append({
            "frame": idx, "t": round(idx / fps, 3),
            "frame_score": None if score is None else round(float(score), 4), "band": band,
            "session": (S["sum"] / S["cnt"]) if S["cnt"] else None,
            "raw": raw_mean, "n_people": n_people,
            "mask": counts["mask"], "incorrect": counts["incorrect"], "no_mask": counts["no_mask"],
            "held": len(missed), "no_people": bool(no_people)})

        S["processed"] += 1
        S["frame_idx"] = idx
        S["elapsed"] = time.perf_counter() - t_run

        # ---- paint ----
        now = time.perf_counter()
        banner = (score, band, n_people) if banner_on else None
        S["last_img"] = annotate(frame, active, per_person, tracker, overlay, banner)
        show_video(ph, S)
        render_fast(ph, S)
        if now - last_slow >= 0.9:
            render_slow(ph, S)
            last_slow = now

        # ---- skip frames / pace to video speed ----
        for _ in range(stride - 1):
            if not cap.grab():
                break
        idx += stride
        if realtime:
            wait = (stride / fps) - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    cap.release()
    S["status"] = "finished"
    S["elapsed"] = time.perf_counter() - t_run

    try:
        pd.DataFrame(S["rows"]).to_csv(alert_log.parent / "dashboard_frames.csv", index=False)
    except OSError:
        pass
    render_fast(ph, S)
    render_slow(ph, S, static=True)
    if tmp_dir:
        try:
            Path(video_path).unlink(missing_ok=True)
            Path(tmp_dir).rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
