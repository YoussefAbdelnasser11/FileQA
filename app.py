import streamlit as st
import tempfile
import os
import torch
from rag_core import RAGSystem

#  إعداد واجهة الصفحة
st.set_page_config(
    page_title="FileQA - RAG System",
    page_icon="📄",
    layout="wide"
)

#  العنوان الرئيسي مع ستايل
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        📄 FileQA: اسأل مستنداتك بسهولة
    </h1>
    <p style='text-align: center; color: #566573; font-size:18px;'>
        نظام ذكي يعتمد على RAG للإجابة على أسئلتك اعتمادًا على ملفات PDF.
    </p>
    """,
    unsafe_allow_html=True
)

#  شريط جانبي للإعدادات
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/pdf.png", use_column_width=True)
    st.header("⚙️ الإعدادات")

    # رفع ملف PDF
    uploaded_file = st.file_uploader("📤 تحميل ملف PDF", type="pdf")

    # اختيار النموذج
    st.subheader(" إعدادات النموذج")
    model_option = st.selectbox(
        "اختر النموذج اللغوي",
        ["mistralai/Mistral-7B-v0.1", "microsoft/DialoGPT-medium"]
    )

    # عرض الجهاز
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.success(f"💻 الجهاز المستخدم: {device_name}")

#  تهيئة النظام وحالة الجلسة
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "processed" not in st.session_state:
    st.session_state.processed = False

#  معالجة الملف
if uploaded_file is not None and not st.session_state.processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        num_chunks = st.session_state.rag_system.process_document(tmp_path)
        st.session_state.processed = True
        st.success(f"✅ تم معالجة المستند بنجاح! (تم إنشاء {num_chunks} جزءًا).")
    except Exception as e:
        st.error(f"❌ خطأ أثناء معالجة المستند: {str(e)}")
    finally:
        os.unlink(tmp_path)

#  واجهة الأسئلة
if st.session_state.processed:
    st.markdown("## ❓ اطرح سؤالك حول المستند")
    question = st.text_input("✍️ اكتب سؤالك هنا:", placeholder="مثال: ما أهم النتائج في هذا المستند؟")

    if st.button(" الحصول على الإجابة") and question:
        with st.spinner("🔍 جاري البحث عن الإجابة..."):
            try:
                answer, context = st.session_state.rag_system.generate_answer(question)
                
                st.markdown("### ✅ الإجابة:")
                st.success(answer)

                with st.expander(" عرض النص المستخدم للإجابة"):
                    st.write(context)
            except Exception as e:
                st.error(f"⚠️ خطأ أثناء توليد الإجابة: {str(e)}")
else:
    st.info("📥 يرجى تحميل ملف PDF للبدء.")

#  تذييل أنيق
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
          تم تطوير <b>FileQA</b> باستخدام 
        <span style="color:#2E86C1;">Streamlit</span>, 
        <span style="color:#27AE60;">Transformers</span>, 
        <span style="color:#D35400;">FAISS</span>, 
        و <span style="color:#8E44AD;">PyPDF2</span>.
    </div>
    """,
    unsafe_allow_html=True
)
