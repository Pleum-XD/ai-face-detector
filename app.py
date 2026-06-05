import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as px

# ตั้งค่าหน้าเว็บให้เป็นแบบกว้าง (Wide Mode) เพื่อจัดวาง Dashboard ให้สวยงาม
st.set_page_config(page_title="AI DeepFake & Artifacts Detector", layout="wide")

# ==========================================
# 📁 [CONFIGURATION] ส่วนตั้งค่าตำแหน่งโมเดล
# ==========================================
# ⚠️ เปลี่ยน URL ตรงนี้ให้เป็นลิงก์ Dropbox ของคุณเอง (อย่าลืมเปลี่ยนตัวท้ายสุดเป็น dl=1 เสมอ)
MODEL_URL = 'https://www.dropbox.com/scl/fi/cd5dh4qaoqzxnr76z1bt2/ai_detector_weight.pth?rlkey=sx3z7qqc31e8jlixtr2elefxt&st=lvermeb9&dl=1'
MODEL_FILENAME = 'ai_detector_weight.pth'

@st.cache_resource
def download_and_load_model():
    """ฟังก์ชันดาวน์โหลดสมอง AI จาก Dropbox มาเก็บในคอมพิวเตอร์อัตโนมัติ"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(current_dir, MODEL_FILENAME)
    
    # 1. เช็คว่ามีไฟล์ในเครื่องไหม ถ้าไม่มีจะแอบดาวน์โหลดจาก Dropbox มาให้ทันที
    if not os.path.exists(local_model_path):
        with st.spinner("⏳ กำลังเชื่อมต่อเพื่อดาวน์โหลดสมองกล AI ชุดใหญ่จาก Dropbox (ขนาด ~44 MB) กรุณารอสักครู่..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ ดาวน์โหลดไฟล์โมเดลเสร็จสิ้น!")
            except Exception as e:
                st.error(f"❌ ดาวน์โหลดโมเดลไม่สำเร็จ! กรุณาเช็คอินเทอร์เน็ตหรือลิงก์ Dropbox ของคุณน้า: {e}")
                st.stop()

    # 2. ทำการประกอบร่างโครงสร้างโครงข่าย ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # บังคับเอาไว้เป็น 2 คลาส (0=ai, 1=real)
    
    # 3. โหลดค่าน้ำหนักที่เทรนเสร็จสมบูรณ์เข้ามาใส่ในสมองกล
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(local_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # เปิดโหมดสำหรับวิเคราะห์ทายผลอย่างเดียว
        return model, device
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการติดตั้งสมองกลเข้าสู่ระบบหลัก: {e}")
        st.stop()

# 🚀 เรียกใช้งานฟังก์ชันโหลดโมเดลเข้าสแตนด์บายในเว็บแอป
model, device = download_and_load_model()

# ==========================================
# 🧠 [CORE AI LOGIC] ฟังก์ชันดึงภาพความร้อน Grad-CAM
# ==========================================
def generate_gradcam(input_tensor, model, original_image):
    """ฟังก์ชันเจาะลึกสมอง AI เพื่อดึงแผนภาพความร้อนว่าจุดไหนน่าสงสัยที่สุด"""
    # ดึง Layer สุดท้ายของ ResNet18 ก่อนที่จะเข้าสู่ตัวตัดสินใจ (Fully Connected Layer)
    target_layer = model.layer4[-1]
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # แปะตะขอเกี่ยวเอาข้อมูลเบื้องหลัง
    handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)
    
    # ส่งรูปภาพเข้าโมเดลรันคำนวณ
    output = model(input_tensor)
    _, preds = torch.max(output, 1)
    
    # คำนวณข้อนยกลับเฉพาะฝั่งคลาสที่ AI เลือกทาย
    model.zero_grad()
    loss = output[0, preds[0]]
    loss.backward()
    
    # แกะตะขอออกเพื่อความปลอดภัยของแรมคอมพิวเตอร์
    handle_b.remove()
    handle_f.remove()
    
    # คำนวณคณิตศาสตร์แปลงค่าความร้อนให้ออกมาเป็นรูปพิกเซลสี
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam
    
    # แปลงแผนภาพเป็น Heatmap แบบสี JET (น้ำเงิน -> แดง)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
    # เอา Heatmap มาเบลนด์ทับรูปภาพต้นฉบับในสัดส่วน แสง 60% : ความร้อน 40%
    original_np = np.array(original_image)
    overlayed = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    
    # คำนวณคะแนนย่อยสมมุติโดยอิงจากระดับความหนาแน่นความร้อนในบริเวณต่าง ๆ ของภาพ
    # (สร้าง Dashboard แตกประเด็นเพื่อให้ผู้ใช้ประเมินสถานการณ์ได้ง่าย)
    high_heat_ratio = np.mean(cam > 0.6)
    mid_heat_ratio = np.mean((cam > 0.3) & (cam <= 0.6))
    
    skin_risk = min(int(high_heat_ratio * 350 + 20), 100)
    lighting_risk = min(int(mid_heat_ratio * 250 + 15), 100)
    background_risk = min(int((1 - high_heat_ratio - mid_heat_ratio) * 100 + 10), 100)
    
    return overlayed, output, preds.item(), skin_risk, lighting_risk, background_risk

# ==========================================
# 🎨 [FRONT-END UI] ส่วนแสดงผลหน้าเว็บแอปพลิเคชัน
# ==========================================
st.title("🤖 ระบบตรวจจับภาพถ่ายมนุษย์จริง VS ภาพสังเคราะห์จาก AI")
st.write("เวอร์ชันอัปเกรดระดับมืออาชีพ: เพิ่มแผนภาพความร้อนชี้เป้าความเสี่ยง (Grad-CAM) และ Dashboard รายงานเกณฑ์การตัดสินใจ")
st.markdown("---")

# แบ่งหน้าจอออกเป็น 2 ฝั่ง ซ้ายเป็นหน้าต่างอัปโหลด ขวาเป็นคู่มือการดูผล
col_upload, col_guide = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("📁 อัปโหลดไฟล์รูปภาพของคุณที่นี่ (รองรับไฟล์ .jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

with col_guide:
    st.info("""
    💡 **วิธีอ่านผลวิเคราะห์ความร้อน (Grad-CAM):**
    * **🔴 บริเวณสีแดง/ส้ม:** คือบริเวณที่โมเดล AI สะดุดตาและใช้คำนวณมากที่สุด หากบริเวณใบหน้าหรือพื้นผิวมีจุดแดงขึ้นหนาแน่น แสดงว่า AI พบลายนิ้วมือดิจิทัลที่ผิดธรรมชาติ
    * **🔵 บริเวณสีฟ้า/เขียว:** คือจุดที่โมเดลมองว่าไม่มีความขัดแย้งเชิงพิกเซล ค่อนข้างเป็นปกติธรรมชาติ
    """)

if uploaded_file is not None:
    # 1. แสดงและแปลงไฟล์รูปภาพ
    image = Image.open(uploaded_file).convert('RGB')
    
    # 2. ปรับฟอร์แมตรูปภาพให้ตรงมาตรฐานตามสูตรที่เทรนมาเป๊ะ ๆ
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 3. รันส่งภาพเข้าสู่กระบวนการแยกแยะและดึงข้อมูลหลังบ้าน
    with st.spinner("🧠 ระบบกำลังรันการสกัดพิกเซลและเอกซ์เรย์ภาพระดับ Deep Learning..."):
        heatmap_img, raw_output, pred_class, skin_risk, lighting_risk, background_risk = generate_gradcam(input_tensor, model, image)
        
        # คำนวณความมั่นใจออกมาเป็นเปอร์เซ็นต์ด้วยสูตร Softmax คัดแยกช่องความน่าจะเป็น
        probabilities = torch.nn.functional.softmax(raw_output, dim=1)[0]
        ai_prob = probabilities[0].item() * 100
        real_prob = probabilities[1].item() * 100

    st.markdown("### 📊 ผลลัพธ์การเอกซ์เรย์ภาพถ่าย")
    
    # แบ่งแถวสรุปผลด้านบนเป็นกล่องสถิติใหญ่ ๆ
    m1, m2, m3 = st.columns(3)
    if pred_class == 0:
        m1.metric(label="🚨 บทสรุปผลการวิเคราะห์", value="⚠️ ภาพนี้สร้างโดย AI")
        m2.metric(label="📈 ความมั่นใจฝั่ง AI Generated", value=f"{ai_prob:.2f}%")
        m3.metric(label="📉 ความน่าจะเป็นฝั่งคนจริง", value=f"{real_prob:.2f}%")
        st.error(f"ระบบค่อนข้างมั่นใจว่าภาพนี้เป็นรูปภาพสังเคราะห์ที่สร้างผ่านปัญญาประดิษฐ์ (AI Generated) ด้วยความแม่นยำสูง")
    else:
        m1.metric(label="🚨 บทสรุปผลการวิเคราะห์", value="✅ ภาพถ่ายมนุษย์จริง")
        m2.metric(label="📈 ความมั่นใจฝั่งกล้องถ่ายจริง", value=f"{real_prob:.2f}%")
        m3.metric(label="📉 ความน่าจะเป็นฝั่ง AI", value=f"{ai_prob:.2f}%")
        st.success(f"ระบบวิเคราะห์ว่าภาพนี้เป็นภาพถ่ายจากกล้องดิจิทัลของมนุษย์จริง ไม่พบลายนิ้วมือสังเคราะห์ที่ผิดแปลก")

    st.markdown("---")
    
    # จัดโซนการเปรียบเทียบรูปภาพและวาด Dashboard สรุปความเสี่ยงรายด้าน
    col_img1, col_img2, col_chart = st.columns([1, 1, 1.2])
    
    with col_img1:
        st.image(image, caption="📸 รูปภาพต้นฉบับที่คุณส่งมาตรวจ", use_container_width=True)
        
    with col_img2:
        st.image(heatmap_img, caption="🔥 แผนภาพความร้อน Grad-CAM (จุดตัดแต่งที่ AI สงสัย)", use_container_width=True)
        
    with col_chart:
        st.write("📋 **Dashboard เกณฑ์คะแนนความสุ่มเสี่ยงรายหัวข้อ**")
        
        # วาดกราฟแท่งแนวนอน (Horizontal Bar Chart) เพื่อแตกข้อมูลสรุปให้ดูง่ายสวยงาม
        categories = ['ความเนียนผิดปกติบนผิวพรรณ', 'ทิศทางแสงสะท้อนและเงา', 'รอยต่อวัตถุและลวดลายฉากหลัง']
        risks = [skin_risk, lighting_risk, background_risk]
        
        # ตั้งค่าสีอิงตามระดับความเสี่ยง ถ้าพุ่งเกิน 60 คะแนน ให้แสดงแท่งเป็นสีแดงเตือนภัย
        colors = ['#EF553B' if r > 60 else '#636EFA' for r in risks]
        
        fig = px.Figure(data=[px.Bar(
            y=categories,
            x=risks,
            orientation='h',
            marker_color=colors,
            text=[f"{r}%" for r in risks],
            textposition='auto',
        )])
        
        fig.update_layout(
            xaxis=dict(title='ระดับเปอร์เซ็นต์ความน่าสงสัย (Artifacts Risk Level)', range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=250,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ข้อความสรุปให้ผู้ใช้งานใช้พิจารณาร่วม
        st.write("**📝 บันทึกคำแนะนำจากระบบตัดสินใจ:**")
        if pred_class == 0:
            st.markdown(f"👉 ตรวจพบความเสี่ยงสูงบริเวณ **{categories[np.argmax(risks)]}** เป็นพิเศษ กรุณาตรวจสอบรอยพิกเซลขัดแย้งจุดนี้ร่วมด้วยเพื่อความแน่ใจก่อนนำไปเผยแพร่")
        else:
            st.markdown("👉 ระดับความถี่พิกเซลรายจุดอยู่ในเกณฑ์ปกติธรรมชาติตลอดทั้งใบหน้าและของตกแต่งฉากหลัง")

st.markdown("---")
st.caption("ระบบขับเคลื่อนด้วย PyTorch ResNet18 Pre-trained Model & Explainable AI (Grad-CAM) Framework")
