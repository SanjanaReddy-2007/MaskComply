# MaskComply dashboard v2 - drop-in upgrade

Replaces `src/dashboard.py` and adds `.streamlit/config.toml` (dark theme). Nothing else in the repo changes.

## Install (PowerShell, from the repo folder)
    Expand-Archive -Path $HOME\Downloads\MaskComply_dashboard_v2.zip -DestinationPath . -Force
    pip install -U "streamlit>=1.43"
    streamlit run src/dashboard.py

Or start straight away on a clip (same as before):
    streamlit run src/dashboard.py -- --video <clip> --yolo_weights models/best.pt

## Using it
- Sidebar: pick a clip (or upload one), set confidence / EMA alpha / overlay, press **Start analysis**.
  Settings are applied only when you press Start, so nothing restarts by accident.
- **Stop** freezes the current frame. Then open the *Session summary* tab and download the frame
  as PNG (choose the "Boxes only" overlay first for a clean report figure).
- Session summary also exports per-frame scores and alerts as CSV.
- `outputs/alert_log.csv` (same format as before) and `outputs/dashboard_frames.csv` are written automatically.
- Tip on a CPU-only PC: lower "Inference size" to 640 or set "Process every Nth frame" to 2-3.
