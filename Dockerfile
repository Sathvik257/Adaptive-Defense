FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8501 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements-host.txt .
RUN pip install --no-cache-dir -r requirements-host.txt
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser adaptive_defense ./adaptive_defense
COPY --chown=appuser:appuser assets ./assets
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser runs/demo ./runs/demo
COPY --chown=appuser:appuser dashboard.py serve.py ./
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8501')+'/_stcore/health',timeout=4)"
CMD ["python", "serve.py"]
