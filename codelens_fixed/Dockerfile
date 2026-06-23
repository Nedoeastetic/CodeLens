FROM python:3.12-slim

WORKDIR /app

# Зависимости — слой кешируется отдельно
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходники и датасет
COPY index.py app.py score.py ./
COPY eval_questions.json sample_queries.txt ./
COPY codebase_python/ ./codebase_python/

EXPOSE 8082

# При старте: индексируем (если chroma_db пуста), затем запускаем UI.
# --fetch-repos скачает open-source репозитории при первом запуске.
CMD ["sh", "-c", \
  "[ -d chroma_db ] || python index.py ./codebase_python/gymhero --fetch-repos && \
   streamlit run app.py \
     --server.address=0.0.0.0 \
     --server.port=8082 \
     --server.headless=true"]
