#!/usr/bin/env python3
"""One short, non-thinking Ollama JSON request to check the local server.

No dataset, secrets, Neo4j, or LLM prediction caches are used or modified.
"""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--num-predict', type=int, default=96)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path('.env'), override=False)
    base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    model = os.getenv('OLLAMA_MODEL', 'qwen3:4b').removeprefix('ollama/')
    url = base + '/api/chat'
    request = {
        'model': model,
        'messages': [
            {'role':'system', 'content':'Respond ONLY with one JSON object with prediction and reason fields.'},
            {'role':'user', 'content':'Previous weekly counts: 5, 5, 5. Predict next direction: increase, flat, or revision.'},
        ],
        'stream':False,
        'think':False,
        'format':'json',
        'options':{'temperature':0,'num_predict':args.num_predict,'num_ctx':1024},
    }
    print(f'Probe: model={model}, server={base}, timeout={args.timeout}s, num_predict={args.num_predict}',flush=True)
    req = Request(url, method='POST', data=json.dumps(request).encode(),
                  headers={'Content-Type':'application/json'})
    started = time.monotonic()
    try:
        with urlopen(req, timeout=args.timeout) as response:
            obj=json.loads(response.read())
    except HTTPError as exc:
        print(f'Ollama HTTP error {exc.code}: {exc.read(400).decode(errors="replace")}')
        raise SystemExit(2)
    except (URLError, TimeoutError) as exc:
        print(f'Ollama could not complete probe: {exc}')
        raise SystemExit(2)
    elapsed=time.monotonic()-started
    content=obj.get('message',{}).get('content','')
    print(json.dumps({'elapsed_seconds':round(elapsed,2),
                      'done_reason':obj.get('done_reason'),
                      'eval_count':obj.get('eval_count'),
                      'response':content[:500]},ensure_ascii=False,indent=2))
    try:
        result=json.loads(content)
        if result.get('prediction') not in ('flat','increase','revision'):
            raise ValueError('prediction missing/invalid')
    except (ValueError,AttributeError) as exc:
        print('Probe returned an invalid prediction JSON:',exc)
        raise SystemExit(3)
    print('Ollama JSON probe passed.')


if __name__=='__main__':
    main()
