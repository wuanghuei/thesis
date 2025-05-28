import streamlit as st
import cv2, math, tempfile, scipy.io as sio, os
from pathlib import Path

st.set_page_config(page_title="MERL Dashboard (Full + 224)", layout="wide")
st.title("MERL Shopping: Video Player & Dataset EDA")

# ==== CÀI ĐẶT THEO MÁY BẠN ====
LABEL_DIR = Path(r"D:\Fpt\Spring 2025\AIP491_Resit\Labels_MERL_Shopping_Dataset")
TARGET_SIZE = (224, 224)
ACTION_NAMES = [
    "Reach To Shelf", "Retract From Shelf", "Hand In Shelf",
    "Inspect Product", "Inspect Shelf"
]

tabs = st.tabs(["Video Player","Dataset EDA"])

with tabs[0]:
    st.header("Video Player (Full + 224×224 + Progress)")
    uploads = st.file_uploader("Upload MP4 files:", type="mp4", accept_multiple_files=True)
    if not uploads:
        st.info("Hãy upload ít nhất 1 file mp4")
    else:
        if "i" not in st.session_state:
            st.session_state.i = 0
        c1,_,c2 = st.columns([1,6,1])
        with c1:
            if st.button("‹ Prev"):
                st.session_state.i = (st.session_state.i-1) % len(uploads)
        with c2:
            if st.button("Next ›"):
                st.session_state.i = (st.session_state.i+1) % len(uploads)

        vf = uploads[st.session_state.i]
        st.markdown(f"**Video {st.session_state.i+1}/{len(uploads)}:** `{vf.name}`")

        # lưu tạm mp4
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vf.read()); tmp.close()
        orig = tmp.name

        # đọc fps + tổng frame
        cap = cv2.VideoCapture(orig)
        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 15
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.release()
        st.write(f"Original: {total} frames @ {fps_orig:.1f} fps")

        # load GT đầy đủ từ .mat
        stem = Path(vf.name).stem.replace("_crop","_label")
        mat = LABEL_DIR/f"{stem}.mat"
        gt_full = [-1]*total
        if mat.exists():
            arr = sio.loadmat(str(mat))["tlabs"]
            for c in range(len(ACTION_NAMES)):
                for s,e in arr[c][0].astype(int):
                    e = min(e, total-1)
                    for j in range(s, e+1):
                        gt_full[j] = c
        else:
            st.warning(f"Không tìm thấy label file: {mat.name}")

        # annotate & play inline
        cap = cv2.VideoCapture(orig)
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out_path = os.path.join(tempfile.gettempdir(), "annotated.webm")
        out = cv2.VideoWriter(out_path, fourcc, fps_orig, TARGET_SIZE)

        pbar = st.progress(0)
        turquoise, white = (255,234,0), (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, TARGET_SIZE)
            cls = gt_full[idx]
            txt_gt = f"{ACTION_NAMES[cls] if cls>=0 else '(None)'}"
            cv2.putText(frame, txt_gt, (5,20), font, font_scale, turquoise, thickness, cv2.LINE_AA)
            txt_fr = f"{idx+1}/{total}"
            (tw,th), _ = cv2.getTextSize(txt_fr, font, font_scale, thickness)
            cv2.putText(frame, txt_fr, (TARGET_SIZE[0]-tw-5, TARGET_SIZE[1]-5), font, font_scale, white, thickness, cv2.LINE_AA)
            out.write(frame)
            pbar.progress((idx+1)/total)

        cap.release()
        out.release()
        pbar.empty()

        # hiển thị video
        data = open(out_path, 'rb').read()
        st.video(data, format="video/webm")

with tabs[1]:
    st.header("MERL Shopping Dataset EDA")
    csv = st.file_uploader("Upload statistics_summary.csv", type="csv")
    if csv:
        df = pd.read_csv(csv)
        st.dataframe(df, use_container_width=True)
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(x="action:N", y="count:Q"),
            use_container_width=True
        )
