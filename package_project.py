"""Build a portable source/demo archive, excluding environments and caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parent
paths=[root/p for p in ["adaptive_defense","docs","tests","data/raw",".streamlit","runs/demo","assets","tools",".github"]]
files=[root/p for p in ["README.md","requirements.txt","requirements-host.txt","dashboard.py","run-demo.ps1","start-dashboard.ps1",".gitignore","package_project.py","Dockerfile",".dockerignore","compose.yaml","serve.py","pyproject.toml"]]
for path in paths:
    files.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
files.extend((root/'web/src').rglob('*.*'))
files.extend(root/'web'/name for name in ['package.json','package-lock.json','tsconfig.json','vite.config.ts','index.html'])
files.extend(root/name for name in ['serve_api.py','requirements-api.txt','Dockerfile.web','compose.web.yaml'])
files.extend(p for folder in ['lab','paper'] for p in (root/folder).rglob('*') if p.is_file() and not {'capture','__pycache__'} & set(p.parts))
with ZipFile(root/"Adaptive-Defense-Project.zip","w",ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file,Path("Adaptive-Defense")/file.relative_to(root))
print("Created Adaptive-Defense-Project.zip")
