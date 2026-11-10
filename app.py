import streamlit as st
import cv2
import numpy as np

# ==========================================
# 0. 页面配置与 CSS 美化 (科技感背景+玻璃拟物化)
# ==========================================
st.set_page_config(page_title="全链路数字图像处理平台", layout="wide", page_icon="🖼️")

st.markdown("""
<style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; justify-content: center; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 10px 10px 0 0; font-size: 16px; font-weight: bold; padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; }
    img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 算法函数区 (全量算法)
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
    return cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# 1.2 相位谱提取 (找回遗漏功能 1)
def phase_extract(img):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    phase = np.angle(dft)
    dft_phase = np.exp(1j * phase)
    img_back = np.fft.ifft2(dft_phase)
    return cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# 1.3 HSV色彩保护增强 (找回遗漏功能 2)
def hsv_enhance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)

# 1.4 CLAHE 自适应增强
def clahe_enhance(img, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab_eq = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# 1.5 频域拉普拉斯锐化
def laplacian_sharpen(img, k=1.0):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# 1.6 形态学顶帽变换
def tophat_correction(img, kernel_size=15):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# 1.7 伪彩色映射
def pseudocolor_map(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

# 1.8 陷波滤波去摩尔纹
def notch_filter(img_gray, notch_x, notch_y, radius):
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    if notch_x > 0 and notch_y > 0:
        cv2.circle(mask, (ccol + notch_x, crow + notch_y), radius, 0, -1)
        cv2.circle(mask, (ccol - notch_x, crow - notch_y), radius, 0, -1)
    f_ishift = np.fft.ifftshift(dft_shift * mask)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), mask

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
# 2. 页面布局与交互
# ==========================================
st.title("🖼️ 全链路数字图像处理与视觉分析平台")
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>基于频域滤波、形态学与多色彩空间的综合性实验系统</p >", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("📸 数据输入区")
    uploaded_file = st.file_uploader("请上传一张测试图片（支持 JPG, PNG）", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 4大顶部标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌙 低照度与频域增强", 
        "🎛️ 摩尔纹消除", 
        "🔬 形态学与自动计数", 
        "🖨️ 压缩与彩色变换"
    ])

    # ---------------- Tab 1: 低照度与频域增强 ----------------
    with tab1:
        st.header("🌙 低照度图像增强与频域特征")
        
        with st.container(border=True):
            st.markdown("#### ⚙️ 算法参数调节")
            # 包含了之前所有6大算法选项
            sub_alg = st.radio("选择增强算法", [
                "同态滤波 (去暗部阴影)", 
                "相位谱提取 (只看轮廓)", 
                "HSV色彩保护 (只提亮不偏色)", 
                "CLAHE自适应增强",
                "频域拉普拉斯锐化",
                "形态学光照校正"
            ], horizontal=True)
            st.markdown("---")
            
            if sub_alg == "同态滤波 (去暗部阴影)":
                col1, col2 = st.columns(2)
                gamma_l = col1.slider("低频增益 (压制阴影)", 0.1, 1.0, 0.5, 0.1)
                gamma_h = col2.slider("高频增益 (提升细节)", 1.0, 3.0, 2.0, 0.1)
            elif sub_alg == "CLAHE自适应增强":
                col1, col2 = st.columns(2)
                clip_l = col1.slider("对比度限制 (Clip Limit)", 1.0, 10.0, 2.0, 0.5)
                tile_s = col2.slider("分块大小 (Tile Size)", 2, 16, 8, 2)
            elif sub_alg == "频域拉普拉斯锐化":
                k_val = st.slider("高频增强系数 (k值)", 0.1, 5.0, 1.0, 0.1)
            elif sub_alg == "形态学光照校正":
                kernel_size = st.slider("结构元素尺寸", 3, 31, 15, 2)
            elif sub_alg in ["相位谱提取 (只看轮廓)", "HSV色彩保护 (只提亮不偏色)"]:
                st.info("该算法无需额外参数，直接查看下方对比结果即可。")

        with st.container(border=True):
            st.markdown("#### 🖼️ 效果对比")
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原始图像", use_column_width=True)
            
            if sub_alg == "同态滤波 (去暗部阴影)":
                res = homomorphic_filter(img_gray, gamma_l, gamma_h)
            elif sub_alg == "相位谱提取 (只看轮廓)":
                res = phase_extract(img_bgr)
            elif sub_alg == "HSV色彩保护 (只提亮不偏色)":
                res = cv2.cvtColor(hsv_enhance(img_bgr), cv2.COLOR_BGR2RGB)
            elif sub_alg == "CLAHE自适应增强":
                res = cv2.cvtColor(clahe_enhance(img_bgr, clip_l, tile_s), cv2.COLOR_BGR2RGB)
            elif sub_alg == "频域拉普拉斯锐化":
                res = laplacian_sharpen(img_bgr, k_val)
            elif sub_alg == "形态学光照校正":
                res = tophat_correction(img_bgr, kernel_size)
                
            c2.image(res, caption=f"处理结果 - {sub_alg}", use_column_width=True, clamp=True)


    # ---------------- Tab 2: 去摩尔纹 ----------------
    with tab2:
        st.header("🎛️ 交互式陷波滤波 (去摩尔纹)")
        
        dft_orig = np.fft.fft2(img_gray)
        dft_orig_shift = np.fft.fftshift(dft_orig)
        magnitude = 20 * np.log(np.abs(dft_orig_shift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        with st.container(border=True):
            st.markdown("#### ⚙️ 频谱操作区")
            colA, colB = st.columns([1, 2])
            colA.image(magnitude, caption="图像频谱 (亮斑代表条纹)", use_column_width=True)
            with colB:
                st.info("💡 拖动下方滑块，相当于在频谱图上「挖去」特定的高频亮点。")
                notch_x = st.slider("X 轴偏离中心距离", 0, img_gray.shape[1]//2, img_gray.shape[1]//8)
                notch_y = st.slider("Y 轴偏离中心距离", 0, img_gray.shape[0]//2, img_gray.shape[0]//8)
                radius = st.slider("陷波器半径 (消除范围)", 1, 50, 10)
        with st.container(border=True):
            st.markdown("#### 🖼️ 效果对比")
            res, mask = notch_filter(img_gray, notch_x, notch_y, radius)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="原始图像", use_column_width=True)
            c2.image(res, caption="滤波处理后", use_column_width=True, clamp=True)


    # ---------------- Tab 3: 自动计数 ----------------
    with tab3:
        st.header("🔬 形态学细胞/颗粒自动计数")
        
        with st.container(border=True):
            st.markdown("#### ⚙️ 形态学与阈值参数")
            col1, col2, col3 = st.columns(3)
            blur_s = col1.slider("高斯去噪尺寸", 1, 15, 5, 2)
            morph_s = col2.slider("形态学核大小", 1, 15, 3, 2)
            area_th = col3.slider("面积过滤阈值", 10, 500, 100, 10)

        with st.container(border=True):
            st.markdown("#### 🖼️ 目标检测结果")
            res_img, morph_img, count = cell_counting(img_bgr, blur_s, morph_s, area_th)
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            st.success(f"✅ **系统提示**：共检测到 **{count}** 个有效目标！")
            
            c1, c2, c3 = st.columns(3)
            c1.image(img_rgb, caption="原始图像", use_column_width=True)
            c2.image(morph_img, caption="二值化分割图", use_column_width=True, clamp=True)
            c3.image(res_rgb, caption="检测标注图", use_column_width=True)


    # ---------------- Tab 4: 压缩与色彩 ----------------
    with tab4:
        st.header("🖨️ 图像压缩模拟 与 🎨 伪彩色映射")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            with st.container(border=True):
                st.markdown("#### 🖨️ JPEG 压缩模拟 (DCT)")
                quality = st.slider("拖动改变压缩质量 Quality", 1, 100, 50)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
                decimg = cv2.imdecode(encimg, 1)
                decimg_rgb = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
                
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="原图", use_column_width=True)
                c2.image(decimg_rgb, caption=f"压缩 Q={quality}", use_column_width=True)

        with col_right:
            with st.container(border=True):
                st.markdown("#### 🎨 伪彩色增强 (灰度映射)")
                st.markdown("*(用于凸显人眼难以分辨的微小灰度差异)*")
                pseudo_res = cv2.cvtColor(pseudocolor_map(img_bgr), cv2.COLOR_BGR2RGB)
                
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="原图", use_column_width=True)
                c2.image(pseudo_res, caption="JET 伪彩色", use_column_width=True)

else:
    st.info("👆 请在页面顶部的【数据输入区】上传一张图片开始探索。")
    st.markdown("""
    ### 💡 快速测试指南：
    1. **低照度增强**：找一张极暗的夜景图或背光人像（建议测试同态滤波和HSV增强）。
    2. **去摩尔纹**：找一张用手机翻拍电脑屏幕（带有波纹条纹）的照片。
    3. **形态学计数**：找一张显微镜下的细胞图，或者散落着一堆硬币的图片。
    4. **压缩与色彩**：随便一张风景图即可，拖动 Quality 滑块看马赛克。
    """)
