import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
import tempfile
import os

class RAGSystem:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.model_name = model_name
        self.device = self.set_device()
        self.tokenizer = None
        self.model = None
        self.model_embeddings = None
        self.index = None
        self.chunks = None
        self.processed = False
        
    def set_device(self):
        """تحديد الجهاز المناسب (GPU/CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def load_language_model(self):
        """تحميل النموذج اللغوي"""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            ).to(self.device)
    
    def extract_text_from_pdf(self, pdf_path):
        """استخراج النص من ملف PDF"""
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        return full_text
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """تقسيم النص إلى أجزاء صغيرة"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def embed_chunks(self, chunks, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        """تحويل الأجزاء النصية إلى متجهات"""
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(chunks, convert_to_numpy=True)
        return model, embeddings
    
    def create_faiss_index(self, embeddings):
        """إنشاء فهرس FAISS للبحث"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    
    def search_index(self, query, k=3):
        """البحث عن الأجزاء الأكثر صلة بالسؤال"""
        query_embedding = self.model_embeddings.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]
    
    def process_document(self, file_path):
        """معالجة المستند بالكامل"""
        text = self.extract_text_from_pdf(file_path)
        
        if len(text.strip()) == 0:
            raise ValueError("لم يتم العثور على نص في ملف PDF")
        
        self.chunks = self.chunk_text(text)
        self.model_embeddings, embeddings = self.embed_chunks(self.chunks)
        self.index = self.create_faiss_index(embeddings)
        self.processed = True
        
        return len(self.chunks)
    
    def generate_answer(self, question):
        """توليد إجابة بناءً على السؤال والمستند"""
        if not self.processed:
            raise ValueError("لم تتم معالجة أي مستند بعد")
        
        # تحميل النموذج اللغوي إذا لم يكن محملاً
        self.load_language_model()
        
        # البحث عن الأجزاء ذات الصلة
        top_chunks = self.search_index(question, k=3)
        
        # إنشاء prompt
        context = " ".join(top_chunks[:2])
        prompt = f"أجب على السؤال التالي بناءً على النص المحدد:\n\nالسؤال: {question}\n\nالنص: {context}\n\nالإجابة:"
        
        # توليد الإجابة
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # استخراج الإجابة فقط
        if "الإجابة:" in answer:
            answer = answer.split("الإجابة:")[-1].strip()
        
        return answer, context
