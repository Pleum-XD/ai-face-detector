import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

# [CONFIGURATION] ส่วนตั้งค่าตำแหน่งโมเดล
# ⚠️ อย่าลืมตรวจสอบลิงก์ Dropbox ของคุณ และเปลี่ยนตัวท้ายสุดเป็น dl=1 เสมอครับ
MODEL_URL = 'https://www.dropbox.com/scl/fi/zqkp6ffp2qkm67zgpcw84/ai_detector_weight.pth?rlkey=wdflmauj7nyku0uti03p6r41t&st=hiqz545v&dl=0'
MODEL_FILENAME = 'ai_detector_weight.pth'

@st.cache_resource
def download_and_load_model():
    """ฟังก์ชันดาวน์โหลดสมอง AI จาก Dropbox มาเก็บในคอมพิวเตอร์อัตโนมัติ"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(current_dir, MODEL_FILENAME)
    
    # 1. เช็คว่ามีไฟล์ในเครื่องไหม ถ้าไม่มีจะดาวน์โหลดจาก Dropbox มาให้ทันที
    if not os.path.exists(local_model_path):
        with st.spinner("⏳ กำลังเชื่อมต่อเพื่อดาวน์โหลดสมองกล AI จาก Dropbox..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ ดาวน์โหลดไฟล์โมเดลเสร็จสิ้น!")
            except Exception as e:
                st.error(f"❌ ดาวน์โหลดโมเดลไม่สำเร็จ! กรุณาเช็คอินเทอร์เน็ตหรือลิงก์ Dropbox ของคุณ: {e}")
                st.stop()

    # 2. ประกอบร่างโครงสร้าง ResNet18 (2 คลาส)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 0=ai, 1=real
    
    # 3. โหลดค่าน้ำหนักที่เทรนเสร็จสมบูรณ์เข้ามาใส่ในสมองกล
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(local_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # เปิดโหมดสำหรับวิเคราะห์ทายผล
        return model, device
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการติดตั้งสมองกลเข้าสู่ระบบหลัก: {e}")
        st.stop()

# เรียกใช้งานฟังก์ชันโหลดโมเดล
model, device = download_and_load_model()

# ส่วนหน้าตาเว็บแอปพลิเคชัน
st.title("🤖 ระบบตรวจจับภาพถ่ายมนุษย์จริง VS ภาพจาก AI")
st.write("เวอร์ชันคลาสสิก: ตรวจสอบความแม่นยำและแสดงผลลัพธ์ทันที")
st.markdown("---")

uploaded_file = st.file_uploader("📁 อัปโหลดไฟล์รูปภาพที่นี่ (รองรับ .jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 รูปภาพที่นำมาตรวจสอบ", width=400)
    
    # แปลงรูปภาพตามมาตรฐานโมเดล
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with st.spinner("🧠 AI กำลังวิเคราะห์รูปภาพ..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, preds = torch.max(outputs, 1)
            
        ai_prob = probabilities[0].item() * 100
        real_prob = probabilities[1].item() * 100

    st.markdown("### 📊 ผลการวิเคราะห์")
    
    if preds.item() == 0:
        st.error(f"🚨 ผลลัพธ์: ภาพนี้สร้างโดย AI (ความมั่นใจ {ai_prob:.2f}%)")
        st.write(f"ℹ️ โอกาสเป็นคนจริง: {real_prob:.2f}%")
    else:
        st.success(f"✅ ผลลัพธ์: ภาพถ่ายมนุษย์จริง (ความมั่นใจ {real_prob:.2f}%)")
        st.write(f"ℹ️ โอกาสเป็นภาพ AI: {ai_prob:.2f}%")
