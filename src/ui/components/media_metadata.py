"""
Authentica AI — Reusable Media Metadata Component.
"""
import streamlit as st


def render_image_metadata(filename: str, size_bytes: int, dimensions: str, format_name: str) -> None:
    size_mb = size_bytes / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_bytes / 1024:.1f} KB"

    html = f"""<div style="background: #0F172A; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 0.8rem; font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
<div style="font-family: monospace; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 0.4rem;">File Information</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
<div><b>Filename:</b> <code>{filename}</code></div>
<div><b>File Size:</b> {size_str}</div>
<div><b>Dimensions:</b> {dimensions}</div>
<div><b>Format:</b> {format_name.upper()}</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_video_metadata(filename: str, size_bytes: int, duration_sec: float, resolution: str, fps: float, total_frames: int) -> None:
    size_mb = size_bytes / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_bytes / 1024:.1f} KB"

    html = f"""<div style="background: #0F172A; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 0.8rem; font-size: 0.84rem; color: #CBD5E1; line-height: 1.6;">
<div style="font-family: monospace; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 0.4rem;">Container Information</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
<div><b>Filename:</b> <code>{filename}</code></div>
<div><b>File Size:</b> {size_str}</div>
<div><b>Duration:</b> {duration_sec:.1f}s</div>
<div><b>Resolution:</b> {resolution}</div>
<div><b>Framerate:</b> {fps:.1f} FPS</div>
<div><b>Total Frames:</b> {total_frames}</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)
