"""Download a manually selected official reference manifest, outside the inference tree.
No automatic gold annotations. Never feed these files into DDPF detection.
"""
import argparse,csv,hashlib,io,json,re,subprocess
from pathlib import Path
from urllib.parse import urlparse
from datetime import date
import requests
from bs4 import BeautifulSoup

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--urls',required=True,help='CSV: source_id,url,publication_date,partition,license');p.add_argument('--gold-dir',required=True);p.add_argument('--allow-host',action='append',default=[]);p.add_argument('--dev-manifest',help='Earlier development source manifest, used to reject exact ID/URL overlap in test');a=p.parse_args();out=Path(a.gold_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'reference_manifest.csv').exists():raise FileExistsError('Reference manifest frozen; choose a different directory')
    with open(a.urls,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    dev=[]
    if a.dev_manifest:
        with open(a.dev_manifest,encoding='utf-8-sig',newline='') as f:dev=list(csv.DictReader(f))
    allowed={'who.int','www.who.int','iris.who.int','ecdc.europa.eu','www.ecdc.europa.eu',*a.allow_host};seen=set();result=[]
    for r in rows:
        id=r['source_id'];url=r['url'];part=r['partition'];date.fromisoformat(r['publication_date'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',id) or id in {'.','..'} or id in seen:raise ValueError('invalid/duplicate source id')
        if part not in {'development','test'}:raise ValueError('invalid partition')
        if part=='test' and any(d.get('source_id')==id or d.get('url')==url for d in dev):raise ValueError('development/test reference overlap')
        seen.add(id);u=urlparse(url)
        if u.scheme!='https' or u.hostname not in allowed:raise ValueError('URL not approved official source: '+url)
        try:
            # Refuse redirects to unapproved host; handle each redirect before making request.
            from urllib.parse import urljoin
            target=url
            for hop in range(6):
                x=requests.get(target,timeout=60,allow_redirects=False,headers={'User-Agent':'PreDiViD research reference acquisition/1.0'})
                if x.is_redirect:
                    target=urljoin(target,x.headers['Location']);uu=urlparse(target)
                    if uu.scheme!='https' or uu.hostname not in allowed:raise ValueError('unapproved redirect');
                    continue
                x.raise_for_status();break
            else:raise ValueError('too many redirects')
            raw=x.content;is_pdf=raw.startswith(b'%PDF');path=out/(id+('.pdf' if is_pdf else '.html'));path.write_bytes(raw)
            if is_pdf:text=subprocess.run(['pdftotext','-layout',str(path),'-'],capture_output=True,check=True).stdout.decode('utf8',errors='replace')
            else:
                soup=BeautifulSoup(raw,'html.parser')
                for el in soup(['script','style','nav','header','footer','noscript']):el.decompose()
                article=soup.select_one('.sf-detail-body-wrapper') or soup.select_one('article') or soup.select_one('main') or soup.body or soup;text=article.get_text(' ',strip=True)
            if len(text.strip())<200:raise ValueError('text extraction too short; review manually')
            dest=out/(id+'.txt');dest.write_text(text,encoding='utf8');h=hashlib.sha256(dest.read_bytes()).hexdigest()
            if part=='test' and any(d.get('sha256')==h for d in dev):raise ValueError('exact text duplicate from development')
            result.append(dict(r,final_url=target,raw_path=path.name,raw_sha256=hashlib.sha256(raw).hexdigest(),text_path=dest.name,sha256=h,status='downloaded_metadata_requires_review'))
        except Exception as e:result.append(dict(r,status='failed',error=type(e).__name__+': '+str(e)[:250]))
    with (out/'reference_manifest.csv').open('w',encoding='utf8',newline='') as f:
        fields=sorted({k for r in result for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
    print(json.dumps({'requested':len(rows),'downloaded':sum(r['status'].startswith('downloaded') for r in result),'status':'review_metadata_and_failures_before_freeze'},indent=2))
if __name__=='__main__':main()
