"""Deterministic, model-free segmentation and alignment. Trace schema version 2."""
from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class AlignmentConfig:
    min_similarity: float = 0.5
    max_merge: int = 3
    max_unit_words: int = 12
    pause_seconds: float = 0.6
    gap_cost: float = 0.6
    merge_cost: float = 0.02
    timing_weight: float = 0.05
    min_coverage: float = 0.9
    review_similarity: float = 0.65
    max_units: int = 250

    def __post_init__(self):
        for name in ('min_similarity', 'gap_cost', 'merge_cost', 'timing_weight',
                     'min_coverage', 'review_similarity'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        for name, upper in [('max_merge', 6), ('max_unit_words', 30), ('max_units', 500)]:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
                raise ValueError(f'{name} must be an integer in [1, {upper}]')
        if not math.isfinite(self.pause_seconds) or self.pause_seconds <= 0:
            raise ValueError('pause_seconds must be positive and finite')

    def to_dict(self):
        return asdict(self)


def segment_words(segments, config):
    """Rebuild clauses across ASR boundaries; retain every word and its provenance.

    Word times are estimates. Never fabricate times for missing ASR words.
    Bounded units are merge candidates, not claims of verified sentences.
    """
    words = []
    for segment in segments:
        if segment['text'].strip() and not segment['words']:
            raise ValueError('ASR returned text without word timestamps')
        if re.sub(r'\s+', '', segment['text']) != re.sub(r'\s+', '', ''.join(w['text'] for w in segment['words'])):
            raise ValueError('ASR segment text differs from timestamped words; review transcription')
        for word in segment['words']:
            if not word['text'].strip():
                continue
            if (not all(math.isfinite(word[k]) for k in ('start', 'end', 'probability'))
                    or word['start'] < 0 or word['end'] < word['start']
                    or not 0 <= word['probability'] <= 1):
                raise ValueError('Invalid ASR word timing or probability')
            if words and word['start'] < words[-1]['start']:
                raise ValueError('ASR words are not chronological')
            words.append({**word, 'segment_id': segment['id'], 'word_id': len(words)})
    units, pending = [], []

    def emit(reason):
        if not pending:
            return
        units.append({'id': len(units), 'text': ''.join(w['text'] for w in pending).strip(),
                      'start': pending[0]['start'], 'end': pending[-1]['end'],
                      'speech_seconds': sum(w['end'] - w['start'] for w in pending),
                      'word_count': len(pending), 'words': list(pending),
                      'asr_probability': sum(w['probability'] for w in pending) / len(pending),
                      'boundary_reason': reason})
        pending.clear()

    for word in words:
        if pending and word['start'] - pending[-1]['end'] >= config.pause_seconds:
            emit('pause')
        pending.append(word)
        if re.search(r'[.!?;,。！？；，]["\u201d\u2019\x27]*$', word['text'].strip()):
            emit('punctuation')
        elif len(pending) >= config.max_unit_words:
            emit('word_limit')
    emit('end_of_audio')
    if len(units) > config.max_units:
        raise ValueError(f'Too many alignment units ({len(units)}); split the recording')
    return units


def spans(units, limit):
    return {(i, length): ' '.join(u['text'] for u in units[i:i + length])
            for i in range(len(units)) for length in range(1, min(limit, len(units)-i)+1)}


def align_units(source, target, similarity, config, trace_candidates=False):
    """Minimum cost monotonic path, weighted by consumed speech mass.

    Each side contributes one unit of total mass. A match costs mass*(1-cosine),
    a gap costs mass*gap_cost. Thus splitting does not create extra reward.
    A small merge cost breaks ties toward smaller, more inspectable spans.
    `similarity(i, ni, j, nj)` returns None for overlong/invalid model inputs.
    """
    n, m = len(source), len(target)
    if max(n, m) > config.max_units:
        raise ValueError('Alignment unit limit exceeded')
    if n*m*config.max_merge**2 > (100000 if trace_candidates else 500000):
        raise ValueError('Candidate budget exceeded; split the recording or reduce max_merge')
    # Use word mass for alignment (stable under speaking rate changes).
    sw = [u['word_count'] / max(1, sum(x['word_count'] for x in source)) for u in source]
    tw = [u['word_count'] / max(1, sum(x['word_count'] for x in target)) for u in target]
    dp = [[math.inf]*(m+1) for _ in range(n+1)]
    back = {}; dp[0][0] = 0.0
    candidates = []
    for i in range(n+1):
        for j in range(m+1):
            if not math.isfinite(dp[i][j]):
                continue
            moves = []
            if i < n:
                moves.append((1, 0, config.gap_cost*sw[i], None, 0.0))
            if j < m:
                moves.append((0, 1, config.gap_cost*tw[j], None, 0.0))
            for ni in range(1, min(config.max_merge, n-i)+1):
                for nj in range(1, min(config.max_merge, m-j)+1):
                    s = similarity(i, ni, j, nj)
                    if s is None:
                        continue
                    if not math.isfinite(s) or not -1.000001 <= s <= 1.000001:
                        raise ValueError('Invalid cosine similarity')
                    s = max(-1.0, min(1.0, s))
                    mass = sum(sw[i:i+ni])+sum(tw[j:j+nj])
                    src_mid = (source[i]['start']+source[i+ni-1]['end'])/2/max(source[-1]['end'], 1e-9)
                    tgt_mid = (target[j]['start']+target[j+nj-1]['end'])/2/max(target[-1]['end'], 1e-9)
                    timing = abs(src_mid-tgt_mid)
                    cost = mass*(1-s+config.merge_cost*(ni+nj-2)+config.timing_weight*timing)
                    if trace_candidates:
                        candidates.append([i, ni, j, nj, s, cost])
                    if s >= config.min_similarity:
                        moves.append((ni, nj, cost, s, timing))
            for ni, nj, cost, s, timing in moves:
                if dp[i][j]+cost < dp[i+ni][j+nj]:
                    dp[i+ni][j+nj] = dp[i][j]+cost
                    back[i+ni, j+nj] = (i, j, ni, nj, cost, s, timing)
    path = []; i, j = n, m
    while i or j:
        pi, pj, ni, nj, cost, s, timing = back[i, j]
        path.append({'source_ids': list(range(pi, i)), 'target_ids': list(range(pj, j)),
                     'similarity': s, 'cost': cost, 'timing_distance': timing,
                     'operation': 'match' if ni and nj else 'source_gap' if ni else 'target_gap'})
        i, j = pi, pj
    path.reverse()
    return {'path': path, 'objective_cost': dp[n][m], 'candidates_recorded': trace_candidates,
            'candidates': candidates, 'candidate_columns': ['source_start', 'source_count',
            'target_start', 'target_count', 'similarity', 'cost']}


def summarize(source, target, chunks, config):
    """Scores are conditional on matched speech; coverage includes every ASR unit."""
    coverage = {}
    for side, units in [('source', source), ('target', target)]:
        matched = {i for c in chunks for i in c[f'{side}_ids']}
        total = sum(u['speech_seconds'] for u in units)
        coverage[side] = (sum(u['speech_seconds'] for u in units if u['id'] in matched)/total
                          if total > 0 else 0.0)
    weights = [sum(source[i]['speech_seconds'] for i in c['source_ids']) for c in chunks]
    total_weight = sum(weights)
    mean = lambda key: (sum(c[key]*w for c, w in zip(chunks, weights))/total_weight
                        if total_weight > 0 else None)
    reasons = []
    if not chunks or total_weight == 0:
        reasons.append('no_scorable_speech')
    if min(coverage.values()) < config.min_coverage:
        reasons.append('insufficient_alignment_coverage')
    if any(c['similarity'] < config.review_similarity for c in chunks):
        reasons.append('weak_alignment_similarity')
    return {'status': 'review_required' if reasons else 'ok', 'review_reasons': reasons,
            'coverage': coverage, 'coverage_basis': 'sum_of_ASR_word_durations',
            'aggregation': 'source_speech_duration_weighted', 'chunks': len(chunks),
            'comet_kiwi': mean('comet_kiwi') if not reasons else None,
            'matched_comet_kiwi': mean('comet_kiwi'), 'metricx_24': mean('metricx_24')}
