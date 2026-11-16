"""Launch GATE headlessly and write raw annotations incrementally, not events."""
import argparse
from ddpf.storyforest.detection_stream import run_gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--output',default='outputs/storyforest/gate_annotations.jsonl');p.add_argument('--gapp',default=None)
    a=p.parse_args()
    print('Processed',run_gate(a.zip,a.output,a.gapp),'new documents (append-only checkpoint).')
if __name__=='__main__':main()
