#!/usr/bin/env python3
"""Create paper-ready provisional tables by combining completed V5/V6/V7 artifacts.
No inference, no model calls. Never upgrades silver annotations into gold.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path


def readj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csvrows(path):
 if not Path(path).exists():return []
 with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def fmt(v,d=3):
 try:return f'{float(v):.{d}f}' if v is not None else 'NA'
 except (ValueError,TypeError):return 'NA'
def tex_escape(s):
 text=str(s)
 for a,b in [('_',r'\_'),('&',r'\&'),('%',r'\%'),('#',r'\#')]:text=text.replace(a,b)
 return text

def binary_class_f1(rows):
 # Macro-F1 over the two gold labels. A spurious 'revision' prediction
 # is still a false negative for its true class, but not an extra gold class.
 result=[]
 for label in ('flat','increase'):
  tp=sum(r['target_label']==label and r['prediction']==label for r in rows)
  fp=sum(r['target_label']!=label and r['prediction']==label for r in rows)
  fn=sum(r['target_label']==label and r['prediction']!=label for r in rows)
  result.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return sum(result)/len(result) if result else None

def build(out,v5_path,v6_path):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 v7_metrics=csvrows(out/'pair_metrics.csv')
 if not v7_metrics:raise RuntimeError('Run the documentary study before paper export')
 v7_manifest=readj(out/'study_manifest.json')
 lines=['# PreDiViD: integrated experimental evidence (V5 + V6 + WHO V7)', '',
  '> DRAFT RESULTS. V7 reference labels are SILVER and NOT independently expert-adjudicated.',
  '> Numeric claims refer to different tasks; comparisons between temporal and documentary tasks are invalid.', '',
  '## A. WHO documentary linking (conditional on 29 curated event mentions)', '',
  '| Model | Method | Positive / labelled pairs | Precision | Recall | F1 | Missing calls |',
  '|---|---|---:|---:|---:|---:|---:|']
 tex=[r'% PreDiViD V7 WHO documentary PILOT -- NOT GOLD, DO NOT CLAIM CAUSALITY.',
      r'\begin{table}[t]',r'\centering\small',
      r'\caption{Exploratory WHO documentary relation identification on manually curated,',
      r'not independently adjudicated silver candidate pairs.}',
      r'\begin{tabular}{llrrrr}',r'\hline',r'Model & Method & N & Precision & Recall & F1 \\ \hline']
 for r in v7_metrics:
  lines.append(f'| {r["model"]} | {r["method"]} | {r["n_positive"]}/{r["n_labeled"]} | {fmt(r["precision"])} | {fmt(r["recall"])} | {fmt(r["f1"])} | {r["technical_missing"]} |')
  tex.append(f'{tex_escape(r["model"])} & {tex_escape(r["method"])} & {r["n_labeled"]} & {fmt(r["precision"])} & {fmt(r["recall"])} & {fmt(r["f1"])} '+r'\\')
 tex.extend([r'\hline',r'\end{tabular}',r'\label{tab:who_silver_links}',r'\end{table}'])
 lines+=['','The linking experiment uses 39 deliberately selected annotated candidate pairs (22 proposed positive).',
 'Do **not** treat missing labels outside these pairs as negatives; the reported F1 is restricted to this silver candidate set.',
 'Actual source-document extraction is evaluated separately as date+event-type slot recall.',
 'The StoryForest `full` condition adds typed metadata and a configuration-driven heuristic guard; it is **not** verified OWL reasoning or BioPropaPhenKG neighborhood retrieval.', '']
 v5=readj(v5_path) if Path(v5_path).exists() else None
 if v5:
  lines+=['## B. Temporal aggregate value prediction (V5)', '',
   f'Full chronological holdout: **{v5["integrity"]["holdout"]}** origin–target pairs; '
   f'KG-covered cohort: **{v5["integrity"]["KG_covered_holdout"]}**.', '',
   '| Method | Cohort | Macro-F1 | Accuracy |','|---|---|---:|---:|']
  names=['M1_persistence','M2_temporal_knn_noKG','M2_temporal_staticKG_knn']
  for r in v5['performance']:
   if r['method'] in names and r['disease']=='ALL' and r['cohort'] in ('full_holdout','kg_eligible'):
    lines.append(f'| {r["method"]} | {r["cohort"]} | {fmt(r["macro_f1"])} | {fmt(r["accuracy"])} |')
  placebo=v5['KG_placebo']
  lines+=['',f'KG negative-control shuffle (99 permutations): original macro-F1 **{fmt(placebo["original_macro_f1"],4)}**, '
    f'shuffled mean **{fmt(placebo["placebo_f1_mean"],4)}**, empirical upper-tail p = **{fmt(placebo["empirical_upper_tail_p_f1"])}**.',
    'Thus improved aggregate accuracy does **not** establish semantic-content benefit; the shuffled profile mean is higher.',
    'In KG-eligible Mpox, the KG model has **zero recall for increases**; its higher accuracy is partly class-prevalence driven.',
    'Contrast p-values are exploratory; Holm-adjusted values from the audit do not support global 5% significance.','']
 else:lines+=['## B. Temporal V5 unavailable: add V5 audit path before manuscript export','']
 v6=readj(v6_path) if Path(v6_path).exists() else None
 if v6:
  pred_csv=Path(v6_path).parent/'predictions_by_size.csv';pred_rows=csvrows(pred_csv)
  lines+=['## C. Model scaling on next-week aggregate direction (V6)', '',
   f'**{v6["n_common_complete_all_models_both_methods"]}** fixed episodes, six Qwen3 sizes, each evaluated with and without KG-context guard.',
   'This task is **not** WHO documentary event linking.','',
   '| Qwen3 | M3 F1 | M4 F1 |', '|---|---:|---:|']
  for mod in v6['models']:
   values=[]
   for meth in ('M3_llm_only','M4_llm_kg_guard'):
    row=next(x for x in v6['metrics'] if x['model']==mod and x['method']==meth
             and x['cohort']=='common_all_models' and x['disease']=='ALL')
    raw=[x for x in pred_rows if x['model']==mod and x['method']==meth]
    value=binary_class_f1(raw) if len(raw)==40 else row['macro_f1']
    values.append(fmt(value,4))
   lines.append(f'| {mod} | {values[0]} | {values[1]} |')
  lines+=['', 'When raw predictions are available, F1 is recomputed consistently over the two observed true labels '
   '(*flat*, *increase*); erroneous *revision* outputs are still penalized without introducing a third gold class.',
   'Across the fixed exploratory sample, scaling model size is not monotonically associated with better performance.','']
 else:lines+=['## C. V6 size manifest unavailable: add V6 manifest path before manuscript export','']
 lines+=['## D. Scientific claims supported and not supported','',
  '**Supported as descriptive:** integrity of split/data; performance on aggregate reported-value direction; '
  'exploratory WHO source-event/relationship pilot; runtime and model-size comparisons on this fixed sample.', '',
  '**Not supported:** causal transmission trajectories, validated incidence, benefit from *correct* KG semantics, '
  'independent StoryForest link gold, end-to-end documentary extraction-to-tree accuracy, complete OWL reasoning, '
  'statistical generalization to unseen outbreaks.','',
  'To cite source data, use the exact WHO document URLs in `source_audit.json`; '
  'a reviewer should adjudicate `human_adjudication.csv` before any *gold standard* claim.','']
 (out/'paper_integrated_results_v7.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (out/'paper_who_silver_table_v7.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
 return {'out_markdown':str(out/'paper_integrated_results_v7.md'),
         'out_latex':str(out/'paper_who_silver_table_v7.tex'),
         'V5_included':v5 is not None,'V6_included':v6 is not None,
         'V6_raw_predictions_recomputed':bool(v6 and (Path(v6_path).parent/'predictions_by_size.csv').exists())}

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--output',default='outputs/realdata/who_storyforest_final_v7')
 ap.add_argument('--v5-audit',default='outputs/realdata/temporal_trajectory_study_v5/final_one_shot_audit.json')
 ap.add_argument('--v6-manifest',default='outputs/realdata/llm_size_study_v6/study_manifest.json')
 a=ap.parse_args();print(json.dumps(build(a.output,a.v5_audit,a.v6_manifest),indent=2))
