import json
import pytest
from fastapi.testclient import TestClient
from adaptive_defense.api.app import create_app


@pytest.fixture
def app(tmp_path):
    artifact = tmp_path/'console.json'
    artifact.write_text(json.dumps({'dataset':'test','projection':'test','policy':'test',
        'flows':[{'id':f'f{i}','score':i} for i in range(22)],'windows':[],
        'metrics':[{'model':'test','fn':2,'fp':3,'recall':.5,'false_positive_rate':.1}],
        'truth':[{'label':1}]}))
    return create_app(artifact, tmp_path/'reviews.db')


def test_budget_and_isolation(app):
    a,b=TestClient(app),TestClient(app)
    assert a.get('/api/reviews').status_code==401
    a.post('/api/session'); b.post('/api/session')
    for i in range(20):
        assert a.post(f'/api/reviews/f{i}').status_code==200
    assert a.post('/api/reviews/f0').status_code==200
    assert a.post('/api/reviews/f20').status_code==409
    assert a.get('/api/reviews').json()['remaining']==0
    assert b.get('/api/reviews').json()['used']==0
    assert b.put('/api/reviews/f0',json={'label':'attack'}).status_code==404
    assert a.put('/api/reviews/f0',json={'label':'attack'}).json()['model_updated'] is False
    assert a.post('/api/reviews/unknown').status_code==404


def test_no_truth_and_costs(app):
    c=TestClient(app)
    assert 'truth' not in c.get('/api/experiment').json()
    result=c.post('/api/costs',json={'missed_attack':10,'false_alarm':.5}).json()
    assert result['results'][0]['cost']==21.5
    assert c.post('/api/session',headers={'origin':'https://untrusted.example'}).status_code==403
    with c.websocket_connect('/ws/replay',headers={'origin':'http://testserver'}) as ws:
        for bad in ([],{'index':True},{'index':-1}):
            ws.send_json(bad)
            assert 'error' in ws.receive_json()
        ws.send_json({'index':0})
        assert ws.receive_json()['flow']=={'id':'f0','score':0}
