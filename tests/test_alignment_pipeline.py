import json
from types import SimpleNamespace

import numpy as np
import pytest

from API.alignment import AlignmentConfig, align_units, segment_words, summarize
from API import eval_core as core


def unit(i, text='word', duration=1):
    return {'id': i, 'text': text, 'start': i, 'end': i+duration,
            'speech_seconds': duration, 'word_count': len(text.split())}


def raw(texts):
    t = 0
    result = []
    for i, text in enumerate(texts):
        words = []
        for w in text.split():
            words.append({'text': ' '+w, 'start': t, 'end': t+.25, 'probability': .95})
            t += .25
        result.append({'id': i, 'text': text, 'words': words})
    return result


def test_reconstructs_across_asr_boundary_and_splits_inside_segment():
    units = segment_words(raw(['What time is', 'it? The empire is old.']), AlignmentConfig())
    assert [u['text'] for u in units] == ['What time is it?', 'The empire is old.']
    assert [w['word_id'] for u in units for w in u['words']] == list(range(8))
    assert units[0]['words'][-1]['segment_id'] == 1


@pytest.mark.parametrize('split,merged', [(.6, .95), (.55, .99)])
def test_better_merged_match_beats_fragmentation(split, merged):
    units = [unit(0), unit(1)]
    def sim(i, ni, j, nj):
        return merged if ni == nj == 2 else split if ni == nj == 1 and i == j else 0
    result = align_units(units, units, sim, AlignmentConfig(timing_weight=0), True)
    assert len(result['path']) == 1
    assert result['path'][0]['source_ids'] == [0, 1]
    assert result['path'][0]['similarity'] == merged


def test_one_to_many_and_explicit_omission():
    source = [unit(0, 'hello world'), unit(1, 'omitted')]
    target = [unit(0, 'hola'), unit(1, 'mundo')]
    result = align_units(source, target, lambda i, ni, j, nj: .95 if (i,ni,j,nj)==(0,1,0,2) else 0,
                         AlignmentConfig())
    assert [s['operation'] for s in result['path']] == ['match', 'source_gap']
    assert result['path'][0]['target_ids'] == [0, 1]


def test_unrelated_or_empty_audio_has_no_matches():
    for source, target in [([], []), ([unit(0)], []), ([unit(0)], [unit(0)])]:
        path = align_units(source, target, lambda *args: .1, AlignmentConfig())['path']
        assert all(s['operation'] != 'match' for s in path)


def test_duration_mean_and_coverage_gate():
    units = [unit(0, duration=1), unit(1, duration=9)]
    chunks = [{'source_ids':[i], 'target_ids':[i], 'similarity':.9,
               'comet_kiwi':s, 'metricx_24':1} for i,s in enumerate([.2, .8])]
    result = summarize(units, units, chunks, AlignmentConfig())
    assert result['comet_kiwi'] == pytest.approx(.74)
    omitted = summarize(units, units, chunks[:1], AlignmentConfig())
    assert omitted['coverage']['source'] == pytest.approx(.1)
    assert omitted['comet_kiwi'] is None
    assert omitted['matched_comet_kiwi'] == .2


def test_weak_alignment_requires_review():
    result = summarize([unit(0)], [unit(0)], [{'source_ids':[0], 'target_ids':[0],
        'similarity':.51, 'comet_kiwi':.9, 'metricx_24':1}], AlignmentConfig())
    assert result['status'] == 'review_required'
    assert result['comet_kiwi'] is None


@pytest.mark.parametrize('config', [{'min_similarity':float('nan')}, {'max_merge':0}, {'max_unit_words':31}])
def test_invalid_parameters(config):
    with pytest.raises(ValueError): AlignmentConfig(**config)


def test_missing_word_timing_fails_explicitly():
    with pytest.raises(ValueError, match='without word timestamps'):
        segment_words([{'text':'speech', 'words':[]}], AlignmentConfig())


def test_asr_requests_word_times_and_language(monkeypatch):
    captured = {}
    def transcribe(path, **kwargs):
        captured.update(kwargs)
        word = SimpleNamespace(word=' Hola.', start=0., end=1., probability=.9)
        segment = SimpleNamespace(text='Hola.', start=0., end=1., avg_logprob=-.1,
                                  no_speech_prob=.01, words=[word])
        return iter([segment]), SimpleNamespace(language='es', language_probability=.99, duration=1.)
    monkeypatch.setattr(core, 'get_whisper', lambda: SimpleNamespace(transcribe=transcribe))
    result = core.transcribe_audio('fake.wav', 'es')
    assert captured['language'] == 'es' and captured['word_timestamps'] is True
    assert result['segments'][0]['words'][0]['probability'] == .9


def test_pipeline_trace_roundtrip_and_replay(monkeypatch, tmp_path):
    source, target = tmp_path/'en.wav', tmp_path/'es.wav'
    source.write_bytes(b'english'); target.write_bytes(b'spanish')
    monkeypatch.setattr(core, 'transcribe_audio', lambda *args: {'segments':raw(['Hello world.']), 'language':'en'})
    monkeypatch.setattr(core, '_encode_spans', lambda units, cfg: {(0,1):np.array([1.,0.])})
    monkeypatch.setattr(core, 'cometeval_batch', lambda pairs:[.82]*len(pairs))
    monkeypatch.setattr(core, 'metricx_eval_batch', lambda pairs:[1.2]*len(pairs))
    monkeypatch.setattr(core, 'sentiment_batch', lambda texts:[{'label':'Neutral'} for _ in texts])
    report = json.loads(json.dumps(core.evaluate_pair(str(source), str(target), debug=True), allow_nan=False))
    assert report['summary']['comet_kiwi'] == .82
    assert report['parameters']['max_merge'] == 3
    assert report['provenance']['source_sha256'] != report['provenance']['target_sha256']
    table = {tuple(row[:4]):row[4] for row in report['trace']['alignment']['candidates']}
    units = report['trace']['units']
    replay = align_units(units['source'], units['target'], lambda *key:table.get(key), AlignmentConfig())
    assert replay['path'] == report['trace']['alignment']['path']
    import subprocess
    import sys
    saved = tmp_path/'trace.json'
    saved.write_text(json.dumps(report), encoding='utf-8')
    cli = subprocess.run([sys.executable, '-m', 'API.debug_evaluation', '--replay', str(saved)],
                         capture_output=True, text=True, check=True)
    assert json.loads(cli.stdout)['replay'] == 'matched'
    monkeypatch.setattr(core, 'cometeval_batch', lambda pairs:[])
    with pytest.raises(ValueError, match='expected 1 finite scores'):
        core.evaluate_pair(str(source), str(target))


def test_model_truncation_is_rejected(monkeypatch):
    tokenizer = SimpleNamespace()
    class Tokenizer:
        model_max_length = 512
        def __call__(self, text, **kwargs): return {'input_ids':list(range(300))}
    monkeypatch.setattr(core, 'get_comet', lambda:SimpleNamespace(encoder=SimpleNamespace(tokenizer=Tokenizer())))
    with pytest.raises(ValueError, match='token budget'):
        core.cometeval_batch([{'src':'long', 'mt':'long'}])


def test_candidate_budget_is_bounded():
    units = [unit(i) for i in range(110)]
    with pytest.raises(ValueError, match='Candidate budget'):
        align_units(units, units, lambda *args: 1., AlignmentConfig(), True)


def test_word_limit_and_pause_preserve_all_content():
    segments = raw(['one two three four'])
    segments[0]['words'][2]['start'] += 2
    segments[0]['words'][2]['end'] += 2
    segments[0]['words'][3]['start'] += 2
    segments[0]['words'][3]['end'] += 2
    units = segment_words(segments, AlignmentConfig(max_unit_words=3))
    assert [u['text'] for u in units] == ['one two', 'three four']
    assert units[0]['boundary_reason'] == 'pause'
    assert sum(u['word_count'] for u in units) == 4


def test_asr_text_mismatch_is_not_silently_scored():
    segments = raw(['hello'])
    segments[0]['text'] = 'hello missing'
    with pytest.raises(ValueError, match='differs'):
        segment_words(segments, AlignmentConfig())


def test_metricx_batches_have_distinct_files_and_timeout(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(core, '_ensure_metricx_repo', lambda: None)
    seen = []
    def run(command, **kwargs):
        path = Path(command[command.index('--input_file')+1])
        assert json.loads(path.read_text())['source'] == 'source'
        seen.append(str(path))
        assert kwargs['timeout'] == 600
        Path(command[command.index('--output_file')+1]).write_text('{"prediction": 1.5}\n')
    monkeypatch.setattr(core.subprocess, 'run', run)
    for _ in range(2):
        assert core.metricx_eval_batch([{'src':'source','mt':'target'}]) == [1.5]
    assert seen[0] != seen[1]
    assert all(not Path(p).exists() for p in seen)


def test_failure_identifies_run_and_stage(monkeypatch, caplog):
    def fail(*args):
        raise RuntimeError('decoder failed')
    monkeypatch.setattr(core, 'transcribe_audio', fail)
    with pytest.raises(RuntimeError, match=r'evaluation=[a-f0-9]+ stage=source_asr: decoder failed'):
        core.evaluate_pair('source.wav', 'target.wav')
    assert 'stage=source_asr failed' in caplog.text
