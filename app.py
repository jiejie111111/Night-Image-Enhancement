import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. 算法函数区 (Algorithm Functions)
# ==========================================

# 1.1 同态滤波
def homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, c=1.0, d0=10):
    img_float = np.float32(img) / 255.0
    img_log = np.log1p(img_float)
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols)
    D2 = (u - crow)**2 + (v - ccol)**2
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * D2 / (d0**2))) + gamma_l
    filtered_dft = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    img_exp = np.expm1(np.real(img_back))
    img_out = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_out

# 1.2 相位谱提取
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

# 1.3 HSV色彩保护增强
def hsv_enhance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    img_out = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return img_out

# 1.4 CLAHE 自适应直方图增强
def clahe_enhance(img, clip_limit=2.0, tile_size=8):
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        img_out = clahe.apply(img)
    return img_out

# 1.5 频域拉普拉斯锐化
def laplacian_sharpen(img, k=1.0):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols)
    H_lap = -4 * (np.pi**2) * ((u - crow)**2 + (v - ccol)**2)
    H = 1 - k * H_lap
    filtered_dft = dft_shift * H
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_dft)))
    img_out = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_out

# 1.6 形态学顶帽变换(光照校正)
def tophat_correction(img, kernel_size=15):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    img_out = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_out

# 1.7 伪彩色映射
def pseudocolor_map(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_out = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    return img_out

# 1.8 陷波滤波去摩尔纹
def notch_filter(img_gray, notch_x, notch_y, radius):
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    
    # 在对称位置挖洞
    if notch_x > 0 and notch_y > 0:
        cv2.circle(mask, (ccol + notch_x, crow + notch_y), radius, 0, -1)
        cv2.circle(mask, (ccol - notch_x, crow - notch_y), radius, 0, -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_out = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return img_out, mask

# 1.9 细胞/颗粒计数
def cell_counting(img_bgr, blur_size, morph_size, area_thresh):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((morph_size, morph_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    result_img = img_bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > area_thresh:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count += 1
    return result_img, morph, count

# ==========================================
# 2. 网页 UI 与 侧边栏导航区
# ==========================================

st.set_page_config(page_title="全链路数字图像处理平台", layout="wide")
st.sidebar.title("🧭 功能导航")
category = st.sidebar.radio("请选择实验模块", 
    ["🌙 低照度图像增强", "🎛️ 频域魔法实验室", "🖨️ 图像压缩与安全", "🔬 形态学与自动计数", "🎨 彩色空间变换"]
)

st.title("🖼️ 全链路数字图像处理与视觉分析平台")
st.markdown("涵盖频域滤波、形态学、图像压缩及色彩变换等核心技术。")
st.markdown("---")

# 统一图片上传
uploaded_file = st.sidebar.file_uploader("📸 请上传测试图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 模块 1：低照度图像增强
    # ---------------------------------------------------------
    if category == "🌙 低照度图像增强":
        st.header("🌙 夜间与暗光图像增强")
        sub_tab = st.radio("选择增强算法", ["同态滤波 (Homomorphic)", "相位谱提取 (Phase)", "HSV色彩保护", "CLAHE自适应增强", "伪彩色映射 (Pseudo-color)"], horizontal=True)
        
        if sub_tab == "同态滤波 (Homomorphic)":
            col1, col2 = st.columns([1, 3])
            with col1:
                gamma_l = st.slider("低频增益 (压制阴影)", 0.1, 1.0, 0.5, 0.1)
                gamma_h = st.slider("高频增益 (提升细节)", 1.0, 3.0, 2.0, 0.1)
            with col2:
                res = homomorphic_filter(img_gray, gamma_l, gamma_h)
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="原图", use_column_width=True)
                c2.image(res, caption="处理后", use_column_width=True, clamp=True)
                
        elif sub_tab == "相位谱提取 (Phase)":
            res = phase_extract(img_bgr)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原图", use_column_width=True)
            c2.image(res, caption="相位谱提取 (仅保留轮廓结构)", use_column_width=True, clamp=True)
            
        elif sub_tab == "HSV色彩保护":
            res = hsv_enhance(img_bgr)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原图", use_column_width=True)
            c2.image(res_rgb, caption="HSV 提亮 (保留色彩)", use_column_width=True)
            
        elif sub_tab == "CLAHE自适应增强":
            col1, col2 = st.columns([1, 3])
            with col1:
                clip_l = st.slider("对比度限制", 1.0, 10.0, 2.0, 0.5)
                tile_s = st.slider("分块大小", 2, 16, 8, 2)
            with col2:
                res = clahe_enhance(img_bgr, clip_l, tile_s)
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="原图", use_column_width=True)
                c2.image(res_rgb, caption="CLAHE 局部增强", use_column_width=True)
                
        elif sub_tab == "伪彩色映射 (Pseudo-color)":
            res = pseudocolor_map(img_bgr)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原图", use_column_width=True)
            c2.image(res_rgb, caption="JET 伪彩色", use_column_width=True)

    # ---------------------------------------------------------
    # 模块 2：频域魔法实验室
    # ---------------------------------------------------------
    elif category == "🎛️ 频域魔法实验室":
        st.header("🎛️ 交互式频域滤波")
        st.markdown("**去摩尔纹 (陷波滤波)**：通过手动在频谱图上挖去特定的亮斑，消除图片中的周期性条纹干扰。")
        
        # 计算原图频谱用于展示
        dft_orig = np.fft.fft2(img_gray)
        dft_orig_shift = np.fft.fftshift(dft_orig)
        magnitude = 20 * np.log(np.abs(dft_orig_shift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(magnitude, caption="频谱图 (亮点=条纹方向)", use_column_width=True)
            st.markdown("---")
            notch_x = st.slider("X 轴偏离中心的距离", 0, img_gray.shape[1]//2, img_gray.shape[1]//8)
            notch_y = st.slider("Y 轴偏离中心的距离", 0, img_gray.shape[0]//2, img_gray.shape[0]//8)
            radius = st.slider("陷波器半径 (挖洞大小)", 1, 50, 10)
            
        with col2:
            res, mask = notch_filter(img_gray, notch_x, notch_y, radius)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原图 (带有周期性条纹)", use_column_width=True)
            c2.image(res, caption="陷波滤波后", use_column_width=True, clamp=True)

    # ---------------------------------------------------------
    # 模块 3：图像压缩与安全
    # ---------------------------------------------------------
    elif category == "🖨️ 图像压缩与安全":
        st.header("🖨️ JPEG 压缩质量模拟")
        st.markdown("拖动下方滑块，观察随着**质量因子(Quality)**下降，图像如何产生8x8的块效应和振铃效应。")
        
        quality = st.slider("JPEG 质量因子 (Quality)", 1, 100, 50)
        
        # 模拟JPEG编码与解码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        decimg_rgb = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="无损原图", use_column_width=True)
        c2.image(decimg_rgb, caption=f"压缩模拟 (Quality = {quality})", use_column_width=True)

    # ---------------------------------------------------------
    # 模块 4：形态学与自动计数
    # ---------------------------------------------------------
    elif category == "🔬 形态学与自动计数":
        st.header("🔬 医学/工业细胞颗粒计数")
        st.markdown("结合**Otsu阈值分割**与**形态学开闭运算**，去除噪点并自动标出目标个数。")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            blur_s = st.slider("高斯滤波去噪尺寸", 1, 15, 5, 2)
            morph_s = st.slider("形态学结构元素大小", 1, 15, 3, 2)
            area_th = st.slider("忽略小噪点 (面积阈值)", 10, 500, 100, 10)
            
        with col2:
            res_img, morph_img, count = cell_counting(img_bgr, blur_s, morph_s, area_th)
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            
            st.success(f"✅ 系统自动识别到 **{count}** 个目标！")
            c1, c2, c3 = st.columns(3)
            c1.image(img_rgb, caption="原图", use_column_width=True)
            c2.image(morph_img, caption="形态学处理后的二值图", use_column_width=True, clamp=True)
            c3.image(res_rgb, caption="检测结果", use_column_width=True)

    # ---------------------------------------------------------
    # 模块 5：彩色空间变换
    # ---------------------------------------------------------
    elif category == "🎨 彩色空间变换":
        st.header("🎨 伪彩色增强与特征凸显")
        st.markdown("人类对灰度的分辨力极弱，将微弱的灰度差异映射为彩色，可用于医学X光片、遥感图像分析。")
        
        res = pseudocolor_map(img_bgr)
        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="原图 (灰度)", use_column_width=True)
        c2.image(res_rgb, caption="JET 伪彩色映射", use_column_width=True)

else:
    st.info("👈 请在左侧边栏上传一张测试图片开始体验。")
    st.markdown("💡 **测试建议：**")
    st.markdown("- 测**低照度**：上传一张极暗的人像或夜景。")
    st.markdown("- 测**频域去摩尔纹**：上传一张翻拍电脑屏幕的照片。")
    st.markdown("- 测**计数**：上传一张细胞图、散落的硬币图。")
