import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'API'))
    monkeypatch.setenv('EVAL_DB_PATH', str(tmp_path/'results.db'))
    for name in ('api', 'eval_store'):
        sys.modules.pop(name, None)
    core = importlib.import_module('eval_core')
    from test_alignment_pipeline import raw
    monkeypatch.setattr(core, 'warm_up', lambda: None)
    monkeypatch.setattr(core, 'transcribe_audio', lambda *args: {'segments':raw(['Hello.']), 'language':'en'})
    monkeypatch.setattr(core, '_encode_spans', lambda units, cfg: {(0,1):np.array([1.,0.])})
    monkeypatch.setattr(core, 'cometeval_batch', lambda pairs: [.8]*len(pairs))
    monkeypatch.setattr(core, 'metricx_eval_batch', lambda pairs: [1.]*len(pairs))
    monkeypatch.setattr(core, 'sentiment_batch', lambda texts:[{'label':'Neutral'} for _ in texts])
    app = importlib.import_module('api').app
    with TestClient(app) as c:
        yield c


def test_http_result_persistence_and_parameters(client):
    response = client.post('/evaluate?source_language=en&target_language=es&debug=true&max_merge=2',
                           files={'source_audio':('en.wav',b'en'), 'target_audio':('es.wav',b'es')})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['parameters']['max_merge'] == 2
    assert data['parameters']['target_language'] == 'es'
    assert data['chunks'][0]['similarity'] == 1.
    assert data['trace']['alignment']['candidates_recorded']
    stored = client.get('/evaluations/'+data['id']).json()
    assert stored['result']['trace'] == data['trace']
    assert stored['result']['parameters'] == data['parameters']
    assert stored['source_name'] == 'en.wav'


@pytest.mark.parametrize('query', ['max_merge=0', 'max_merge=7', 'min_similarity=2', 'max_unit_words=31'])
def test_http_invalid_configuration(client, query):
    response = client.post('/evaluate?'+query,
                           files={'source_audio':('en.wav',b'en'), 'target_audio':('es.wav',b'es')})
    assert response.status_code == 422
