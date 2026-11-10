"""Run a traced audio evaluation or replay its alignment without models."""
import argparse
import json
import logging
from pathlib import Path

from .alignment import AlignmentConfig, align_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--source-language', default='en')
    parser.add_argument('--target-language', default='es')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.replay:
        report = json.loads(args.replay.read_text(encoding='utf-8'))
        trace = report['trace']
        candidates = trace['alignment']['candidates']
        if not trace['alignment'].get('candidates_recorded', False):
            parser.error('Replay requires a report generated with debug=true')
        table = {tuple(row[:4]): row[4] for row in candidates}
        config = AlignmentConfig(**{k: v for k, v in report['parameters'].items()
                                    if k in AlignmentConfig.__dataclass_fields__})
        actual = align_units(trace['units']['source'], trace['units']['target'],
                             lambda *key: table.get(key), config)
        if actual['path'] != trace['alignment']['path']:
            raise RuntimeError('Alignment replay differs from the saved path')
        print(json.dumps({'run_id':report['run_id'], 'replay':'matched',
                          'objective_cost':actual['objective_cost']}))
        return
    if not args.source or not args.target or not args.output:
        parser.error('--source, --target and --output are required for an audio run')
    from .eval_core import evaluate_pair
    report = evaluate_pair(str(args.source), str(args.target), source_language=args.source_language,
                           target_language=args.target_language, debug=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'run_id':report['run_id'], 'summary':report['summary']}, indent=2))


if __name__ == '__main__':
    main()
