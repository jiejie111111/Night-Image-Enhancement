import streamlit as st
import cv2
import numpy as np


# ================= 算法函数区 =================

def homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, c=1.0, d0=10):
    img_float = np.float32(img) / 255.0
    img_log = np.log1p(img_float)
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols)
    D2 = (u - crow) ** 2 + (v - ccol) ** 2
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * D2 / (d0 ** 2))) + gamma_l
    filtered_dft = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    img_exp = np.expm1(np.real(img_back))
    img_out = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_out


def phase_extract(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    phase = np.angle(dft)
    dft_phase = np.exp(1j * phase)
    img_back = np.fft.ifft2(dft_phase)
    img_phase = np.abs(img_back)
    img_phase = cv2.normalize(img_phase, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_phase


def hsv_enhance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    img_out = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return img_out


def target_lock(img, blur_size=5, morph_size=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((morph_size, morph_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    img_out = img.copy()
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
    return img_out, count, morph


# ================= 网页 UI 区 =================

st.set_page_config(page_title="夜间图像分析系统", layout="wide")
st.title("🌃 夜间低照度图像增强与分析系统")
st.markdown("结合频域滤波、色彩空间与形态学，完成从图像增强到目标提取的完整闭环。")
st.markdown("---")

st.sidebar.header("📸 上传图片")
uploaded_file = st.sidebar.file_uploader("请选择一张图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 将上传的文件转换为 OpenCV 可以读取的格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 转换为 RGB 用于网页显示 (Streamlit 用 RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 建立四个标签页
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌓 暗部细节复活 (同态滤波)", "👻 相位轮廓提取", "🎨 色彩保护增强 (HSV)", "🎯 目标自动锁定 (形态学)"])

    with tab1:
        st.header("同态滤波：压制低频光照，提升高频细节")
        col1, col2 = st.columns([1, 3])
        with col1:
            gamma_l = st.slider("低频增益 (Gamma L)", 0.1, 1.0, 0.5, 0.1)
            gamma_h = st.slider("高频增益 (Gamma H)", 1.0, 3.0, 2.0, 0.1)
        with col2:
            # 执行算法
            homo_res = homomorphic_filter(img_gray, gamma_l, gamma_h)
            colA, colB = st.columns(2)
            colA.image(img_rgb, caption="原始暗光图", use_column_width=True)
            colB.image(homo_res, caption="同态滤波后", use_column_width=True, clamp=True)

    with tab2:
        st.header("相位谱重构：忽略亮度，只取轮廓")
        st.markdown("当丢弃幅度谱（亮度信息），只保留相位谱（位置信息）时，图像的边缘结构依然清晰。")
        phase_res = phase_extract(img_bgr)
        colA, colB = st.columns(2)
        colA.image(img_rgb, caption="原始图", use_column_width=True)
        colB.image(phase_res, caption="相位谱提取", use_column_width=True, clamp=True)

    with tab3:
        st.header("色彩保护提亮：只动亮度，不动颜色")
        st.markdown("在 HSV 空间中对亮度 V 通道进行直方图均衡化，避免直接处理 RGB 导致颜色失真。")
        hsv_res = hsv_enhance(img_bgr)
        # 转为 RGB 显示
        hsv_res_rgb = cv2.cvtColor(hsv_res, cv2.COLOR_BGR2RGB)
        colA, colB = st.columns(2)
        colA.image(img_rgb, caption="原始图", use_column_width=True)
        colB.image(hsv_res_rgb, caption="HSV 色彩保护增强", use_column_width=True)

    with tab4:
        st.header("机器视觉：自动阈值与形态学目标锁定")
        col1, col2 = st.columns([1, 3])
        with col1:
            blur_s = st.slider("高斯去噪核大小 (奇数)", 1, 15, 5, 2)
            morph_s = st.slider("形态学结构元素大小", 1, 15, 3, 2)
        with col2:
            lock_res, count, morph_img = target_lock(img_bgr, blur_s, morph_s)
            lock_res_rgb = cv2.cvtColor(lock_res, cv2.COLOR_BGR2RGB)

            st.success(f"✅ 识别完成！系统共锁定 **{count}** 个目标区域。")
            colA, colB, colC = st.columns(3)
            colA.image(img_rgb, caption="原始图", use_column_width=True)
            colB.image(morph_img, caption="Otsu 分割与形态学去噪", use_column_width=True, clamp=True)
            colC.image(lock_res_rgb, caption="目标检测结果", use_column_width=True)

else:
    st.info("👈 请在左侧边栏上传一张夜间照片或带有噪声的照片开始体验。")