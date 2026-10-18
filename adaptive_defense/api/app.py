"""Same-origin React hosting, recorded WebSocket replay and budgeted review queue."""
import asyncio
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[2]

class ReviewLabel(BaseModel):
    label: Literal['benign','attack','new-family']

class Costs(BaseModel):
    missed_attack: float=Field(default=10,ge=0,le=100)
    false_alarm: float=Field(default=.5,ge=0,le=100)
    query: float=Field(default=.3,ge=0,le=100)

def create_app(artifact=None, database=None):
    app=FastAPI(title='Adaptive Defense API',version='0.3.0')
    artifact=Path(artifact or ROOT/'runs/demo/console.json')
    database=Path(database or os.environ.get('AD_DATABASE',str(ROOT/'runtime/reviews.sqlite3')))
    database.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(database) as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, budget INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews(session_id TEXT, flow_id TEXT, label TEXT, PRIMARY KEY(session_id,flow_id));''')
    dataset=json.loads(artifact.read_text(encoding='utf-8')) if artifact.exists() else None
    flows={f['id']:f for f in dataset['flows']} if dataset else {}

    def ready():
        if dataset is None:
            raise HTTPException(503,'Generate the console artifact with python tools/export_console.py')

    def identify(request):
        token=request.cookies.get('ad_session','')
        with sqlite3.connect(database) as db:
            row=db.execute('SELECT budget FROM sessions WHERE id=?',(token,)).fetchone()
        if row is None:
            raise HTTPException(401,'Start a review session first.')
        return token,row[0]

    @app.middleware('http')
    async def origin_check(request,call_next):
        if request.method in ('POST','PUT','DELETE'):
            origin=request.headers.get('origin')
            allowed={str(request.base_url).rstrip('/'),'http://127.0.0.1:5173','http://localhost:5173'}
            if origin and origin not in allowed:
                return Response('Cross-origin writes are not permitted',status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/health')
    def health():
        ready()
        return {'status':'ok','mode':'recorded-synthetic-demo'}

    @app.get('/api/experiment')
    def experiment():
        ready()
        return {key:dataset[key] for key in ('dataset','projection','policy','flows','metrics','windows')}

    @app.post('/api/session')
    def session(request:Request,response:Response):
        token=request.cookies.get('ad_session')
        with sqlite3.connect(database) as db:
            found=db.execute('SELECT budget FROM sessions WHERE id=?',(token,)).fetchone() if token else None
            if not found:
                token=secrets.token_urlsafe(32)
                db.execute('INSERT INTO sessions VALUES(?,?)',(token,20))
        response.set_cookie('ad_session',token,httponly=True,samesite='strict',secure=request.url.scheme=='https',max_age=86400)
        return {'budget':20}

    @app.get('/api/reviews')
    def reviews(request:Request):
        token,budget=identify(request)
        with sqlite3.connect(database) as db:
            rows=db.execute('SELECT flow_id,label FROM reviews WHERE session_id=?',(token,)).fetchall()
        return {'budget':budget,'used':len(rows),'remaining':budget-len(rows),
                'reviews':[{'flow_id':row[0],'label':row[1]} for row in rows]}

    @app.post('/api/reviews/{flow_id}')
    def ask(flow_id:str,request:Request):
        ready(); token,budget=identify(request)
        if flow_id not in flows:
            raise HTTPException(404,'Unknown flow')
        with sqlite3.connect(database) as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM reviews WHERE session_id=? AND flow_id=?',(token,flow_id)).fetchone():
                return {'status':'already-queued'}
            used=db.execute('SELECT COUNT(*) FROM reviews WHERE session_id=?',(token,)).fetchone()[0]
            if used>=budget:
                raise HTTPException(409,'Analyst budget exhausted')
            db.execute('INSERT INTO reviews VALUES(?,?,NULL)',(token,flow_id))
        return {'status':'queued'}

    @app.put('/api/reviews/{flow_id}')
    def label(flow_id:str,payload:ReviewLabel,request:Request):
        token,_=identify(request)
        with sqlite3.connect(database) as db:
            changed=db.execute('UPDATE reviews SET label=? WHERE session_id=? AND flow_id=?',(payload.label,token,flow_id)).rowcount
            if not changed:
                raise HTTPException(404,'Queue this flow before labeling it.')
        return {'status':'saved','model_updated':False}

    @app.post('/api/costs')
    def costs(payload:Costs):
        ready()
        return {'results':[{'model':row['model'],'cost':row['fn']*payload.missed_attack+row['fp']*payload.false_alarm,
                           'recall':row['recall'],'false_positive_rate':row['false_positive_rate']} for row in dataset['metrics']],
                'note':'Re-scoring fixed predictions; this does not retrain the agent. Query cost applies only to the review budget.'}

    @app.websocket('/ws/replay')
    async def replay(websocket:WebSocket):
        origin=websocket.headers.get('origin','')
        host=websocket.headers.get('host','')
        if origin not in {f'http://{host}',f'https://{host}','http://127.0.0.1:5173','http://localhost:5173'}:
            await websocket.close(code=1008); return
        await websocket.accept()
        if dataset is None:
            await websocket.close(code=1011); return
        try:
            # Client pull provides deterministic backpressure and pause without queued timers.
            while True:
                command=await websocket.receive_json()
                index=command.get('index') if isinstance(command,dict) else None
                if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(dataset['flows']):
                    await websocket.send_json({'error':'Invalid replay index'}); continue
                await websocket.send_json({'type':'flow','flow':dataset['flows'][index]})
        except (WebSocketDisconnect,ValueError):
            pass

    static=ROOT/'web/dist'
    if static.exists():
        app.mount('/assets',StaticFiles(directory=static/'assets'),name='assets')
        @app.get('/')
        def index():
            return FileResponse(static/'index.html')
    return app

app=create_app()
