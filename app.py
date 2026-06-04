import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# --- 1. ตั้งค่าลิงก์ดาวน์โหลดโมเดลจาก Dropbox ---
MODEL_PATH = 'ai_detector_weight.pth'

# ⚠️ สำคัญมาก: เอาลิงก์ Dropbox ของคุณที่เปลี่ยนตัวท้ายเป็น dl=1 แล้ว มาวางแทนที่ข้อความข้างล่างนี้ครับ
MODEL_URL = 'https://www.dropbox.com/scl/fi/y2ql6p87b0juxsnsenwzq/ai_detector_weight.pth?rlkey=7nmxmiorwflh31f75wbptt1hp&st=0h7jvhtt&dl=1'

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('📦 กำลังดาวน์โหลดสมอง AI จาก Server (จะทำเฉพาะครั้งแรก)...'):
            try:
                # ป้องกันการโดนบล็อกการเชื่อมต่อจาก Server
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success('✅ ดาวน์โหลดโมเดลสำเร็จ!')
            except Exception as e:
                st.error(f'❌ ดาวน์โหลดไม่สำเร็จ: {e}')

# --- 2. โครงสร้างโมเดล (ต้องตรงกับตอนเทรนเป๊ะๆ เป็นแบบ 2 คลาส) ---
def get_model_structure():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # ล็อกไว้ที่ 2 คลาสเหมือนตอนเทรนในคอม
    return model

@st.cache_resource
def load_trained_model():
    download_model()
    if os.path.exists(MODEL_PATH):
        try:
            model = get_model_structure()
            # โหลดค่าน้ำหนักเข้ามาใช้งานบน CPU
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH) # หากไฟล์เสียให้ลบทิ้งเพื่อโหลดใหม่
            st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
            return None
    return None

# --- 3. ส่วนการแสดงผลบนหน้าต่างเว็บ (UI) ---
st.set_page_config(page_title="AI vs Real Human Face Detector", page_icon="🛡️")
st.title("🛡️ AI Human Face Detector")
st.write("อัปโหลดรูปภาพใบหน้าคนเพื่อตรวจสอบว่าเป็นภาพถ่ายจริง หรือสร้างโดย AI")

# เรียกใช้งานโมเดล
model = load_trained_model()

# ปุ่มสำหรับอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกรูปภาพใบหน้าคน...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='รูปภาพที่คุณอัปโหลด', use_container_width=True)
    
    if st.button('🚀 เริ่มต้นการวิเคราะห์ภาพ'):
        if model is not None:
            with st.spinner('🔍 กำลังวิเคราะห์โครงสร้างพิกเซล...'):
                # ปรับแต่งรูปภาพที่คนอัปโหลดเข้ามาให้กลายเป็นสเกลเดียวกับที่ AI เคยเรียนรู้
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = preprocess(image).unsqueeze(0)
                
                # ส่งรูปให้ AI ทายผล
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                
                # แยกคะแนนความน่าจะเป็นออกเป็นเปอร์เซ็นต์ (คลาส 0 = ai, คลาส 1 = real)
                ai_score = prob[0].item() * 100
                real_score = prob[1].item() * 100
                
                # แสดงผลลัพธ์บนหน้าจอ
                st.markdown("### 📊 ผลการวิเคราะห์จากระบบ")
                col1, col2 = st.columns(2)
                col1.metric("โอกาสที่เป็นภาพ AI", f"{ai_score:.2f}%")
                col2.metric("โอกาสที่เป็นรูปถ่ายจริง", f"{real_score:.2f}%")
                
                if ai_score > real_score:
                    st.error(f"⚠️ ผลลัพธ์: ระบบมั่นใจว่าเป็นภาพสร้างจาก **AI** ({ai_score:.1f}%)")
                else:
                    st.success(f"✅ ผลลัพธ์: ระบบมั่นใจว่าเป็น **รูปถ่ายมนุษย์จริง** ({real_score:.1f}%)")
                st.progress(ai_score / 100)
        else:
            st.error("ระบบโมเดลยังไม่พร้อมทำงาน กรุณาเช็คลิงก์ดาวน์โหลดครับ")
