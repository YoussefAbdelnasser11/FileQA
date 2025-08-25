import streamlit as st
import tempfile
import os
from rag_core import RAGSystem

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="نظام RAG للأسئلة حول المستندات",
    page_icon="📄",
    layout="wide"
)

# عنوان التطبيق
st.title("📄 نظام RAG للأسئلة حول المستندات")
st.markdown("""
هذا التطبيق يستخدم تقنية RAG (Retrieval-Augmented Generation) للإجابة على الأسئلة بناءً على محتوى مستند PDF.
قم بتحميل مستند PDF ثم اطرح أسئلتك لتحصل على إجابات دقيقة.
""")

# تهيئة حالة الجلسة
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if "processed" not in st.session_state:
    st.session_state.processed = False

# شريط جانبي للتحميل والإعدادات
with st.sidebar:
    st.header("الإعدادات")
    uploaded_file = st.file_uploader("تحميل ملف PDF", type="pdf")
    
    # إعدادات النموذج
    st.subheader("إعدادات النموذج")
    model_option = st.selectbox(
        "اختر النموذج اللغوي",
        ["mistralai/Mistral-7B-v0.1", "microsoft/DialoGPT-medium"]
    )
    
    # معالجة الجهاز
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"الجهاز المستخدم: {device_name}")

# معالجة الملف المرفوع
if uploaded_file is not None and not st.session_state.processed:
    # حفظ الملف مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # معالجة المستند
        num_chunks = st.session_state.rag_system.process_document(tmp_path)
        st.session_state.processed = True
        st.success(f"تم معالجة المستند بنجاح! تم إنشاء {num_chunks} جزءًا.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة المستند: {str(e)}")
    finally:
        # حذف الملف المؤقت
        os.unlink(tmp_path)

# واجهة الأسئلة إذا تم معالجة المستند
if st.session_state.processed:
    st.subheader("اطرح سؤالاً حول المستند")
    
    question = st.text_input("اكتب سؤالك هنا:", placeholder="متى ولد أندرو سامر؟")
    
    if st.button("الحصول على الإجابة") and question:
        with st.spinner("جاري البحث عن الإجابة..."):
            try:
                answer, context = st.session_state.rag_system.generate_answer(question)
                
                st.subheader("الإجابة:")
                st.success(answer)
                
                with st.expander("عرض النص المستخدم للإجابة"):
                    st.text(context)
            except Exception as e:
                st.error(f"حدث خطأ أثناء توليد الإجابة: {str(e)}")
else:
    st.info("⏳ يرجى تحميل ملف PDF لبدء استخدام التطبيق.")

# تذييل الصفحة
st.markdown("---")
st.markdown("""
تم تطوير هذا التطبيق باستخدام:
- **Streamlit** لواجهة المستخدم
- **Transformers** للنماذج اللغوية
- **FAISS** للبحث المتجهي
- **PyPDF2** لمعالجة مستندات PDF
""")
