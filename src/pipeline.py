"""Reproducible, local recommendation pipeline; see README for protocol."""
from pathlib import Path
import json, hashlib, time, traceback, shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/config.json').read_text())

def save(df, path):
    p = ROOT/path; p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
def load(path): return pd.read_csv(ROOT/path)
def write(path, obj):
    p = ROOT/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=2, default=str)+'\n')

def ingest():
    if CFG['source'] == 'csv':
        for name in ('users','items','interactions'):
            if not (ROOT/f'data/raw/{name}.csv').exists(): raise FileNotFoundError(name)
        return
    if CFG['source'] != 'synthetic': raise ValueError('source must be synthetic or csv')
    rng = np.random.default_rng(CFG['seed']); cats = ['Action','Comedy','Drama','Sci-Fi','Documentary']
    users = pd.DataFrame({'user_id': [f'u{i:03}' for i in range(1,121)], 'age': rng.integers(18,75,120), 'region': rng.choice(['North','South','East','West'],120)})
    items = pd.DataFrame({'item_id':[f'i{i:03}' for i in range(1,101)], 'title':[f'Movie {i:03}' for i in range(1,101)], 'category':[cats[i%5] for i in range(100)]})
    rows=[]
    for user in users.user_id:
        pref = rng.choice(cats,2,replace=False); weights = np.where(items.category.isin(pref),4.,1.); weights /= weights.sum()
        chosen = rng.choice(100, size=35, replace=False, p=weights)
        days = np.sort(rng.choice(300,35,replace=False))
        for j,d in zip(chosen,days):
            rating = int(np.clip(np.rint(rng.normal(4.1 if items.category.iloc[j] in pref else 3.1,0.8)),1,5))
            rows.append((user,items.item_id.iloc[j],rating,(pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=int(d))).isoformat()))
    save(users,'data/raw/users.csv'); save(items,'data/raw/items.csv'); save(pd.DataFrame(rows,columns=['user_id','item_id','rating','timestamp']),'data/raw/interactions.csv')

def validate():
    schemas={'users':['user_id','age','region'],'items':['item_id','title','category'],'interactions':['user_id','item_id','rating','timestamp']}
    valid={}; report={}
    for name, cols in schemas.items():
        df=load(f'data/raw/{name}.csv'); missing=set(cols)-set(df.columns)
        if missing: raise ValueError(f'{name}: missing columns {sorted(missing)}')
        df=df[cols].copy(); reasons=pd.Series('',index=df.index,dtype='object')
        def flag(mask,reason):
            nonlocal reasons
            reasons.loc[mask.fillna(True)] += reason+';'
        required={'users':['user_id'],'items':['item_id','title','category'],'interactions':['user_id','item_id','rating','timestamp']}[name]
        for c in required:
            flag(df[c].isna() | df[c].astype(str).str.strip().eq(''),f'missing_{c}')
        if name in ('users','items'):
            key='user_id' if name=='users' else 'item_id'; flag(df.duplicated(key,keep='first'),'duplicate_id')
        if name=='users':
            age=pd.to_numeric(df.age,errors='coerce'); flag(df.age.notna() & (age.isna() | ~age.between(0,120)),'invalid_age'); df['age']=age
            df['region']=df.region.fillna('Unknown').astype(str).str.strip().replace('','Unknown')
        if name=='interactions':
            rating=pd.to_numeric(df.rating,errors='coerce'); flag(rating.isna() | ~rating.between(1,5),'invalid_rating'); df['rating']=rating
            ts=pd.to_datetime(df.timestamp,errors='coerce',utc=True); flag(ts.isna(),'invalid_timestamp'); df['timestamp']=ts
            flag(df.duplicated(['user_id','item_id','timestamp'],keep='first'),'duplicate_interaction')
            flag(~df.user_id.isin(valid['users'].user_id),'unknown_user'); flag(~df.item_id.isin(valid['items'].item_id),'unknown_item')
        bad=df.loc[reasons.ne('')].copy(); bad['error_reasons']=reasons[reasons.ne('')]; good=df.loc[reasons.eq('')].copy()
        valid[name]=good; save(good,f'data/processed/{name}_valid.csv'); save(bad,f'data/errors/{name}_errors.csv')
        report[name]={'input':len(df),'accepted':len(good),'rejected':len(bad),'reason_counts':reasons[reasons.ne('')].str.split(';').explode().replace('',pd.NA).value_counts().to_dict()}
    write('reports/validation_report.json',report)
    if valid['interactions'].empty: raise ValueError('No valid interactions')

def prepare():
    df=load('data/processed/interactions_valid.csv'); df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
    cutoff=pd.Timestamp(CFG['cutoff'],tz='UTC'); train=df[df.timestamp < cutoff].copy(); test=df[df.timestamp >= cutoff].copy()
    if train.empty or test.empty: raise ValueError('Both time-split partitions must be nonempty')
    save(train,'data/processed/train.csv'); save(test,'data/processed/test.csv')
    write('reports/split.json',{'cutoff_utc':str(cutoff),'train':len(train),'test':len(test),'rule':'train timestamp < cutoff; test timestamp >= cutoff'})

def features():
    tr=load('data/processed/train.csv'); items=load('data/processed/items_valid.csv'); users=load('data/processed/users_valid.csv')
    tr['timestamp']=pd.to_datetime(tr.timestamp,utc=True); cutoff=pd.Timestamp(CFG['cutoff'],tz='UTC')
    u=tr.groupby('user_id').agg(activity_count=('item_id','size'),average_rating=('rating','mean'),last_interaction=('timestamp','max')).reset_index()
    u['days_since_last_interaction']=(cutoff-u.last_interaction).dt.total_seconds()/86400
    pref=tr.merge(items[['item_id','category']],on='item_id').groupby(['user_id','category']).size().reset_index(name='n').sort_values(['user_id','n','category'],ascending=[True,False,True]).drop_duplicates('user_id')
    u=users.merge(u,on='user_id',how='left').merge(pref[['user_id','category']].rename(columns={'category':'preferred_category'}),on='user_id',how='left')
    it=tr.groupby('item_id').agg(popularity=('user_id','size'),average_rating=('rating','mean')).reset_index(); it=items.merge(it,on='item_id',how='left'); it['popularity']=it.popularity.fillna(0).astype(int)
    save(u,'data/features/user_features.csv'); save(it,'data/features/item_features.csv')
    pairs=tr.groupby(['user_id','item_id']).agg(interaction_count=('rating','size'),mean_rating=('rating','mean'),last_interaction=('timestamp','max')).reset_index(); pairs['days_since_last_interaction']=(cutoff-pairs.last_interaction).dt.total_seconds()/86400
    save(pairs,'data/features/interaction_features.csv')
    matrix=pairs.pivot(index='user_id',columns='item_id',values='interaction_count').fillna(0); matrix.to_csv(ROOT/'data/processed/user_item_matrix.csv')

def train():
    pairs=load('data/features/interaction_features.csv'); items=load('data/features/item_features.csv'); users=sorted(pairs.user_id.unique()); ids=sorted(items.item_id.unique())
    ui={u:i for i,u in enumerate(users)}; ii={v:i for i,v in enumerate(ids)}; matrix=np.zeros((len(users),len(ids)))
    for row in pairs.itertuples(): matrix[ui[row.user_id],ii[row.item_id]]=1
    # Truncated SVD of binary implicit feedback; reconstruction ranks unseen items.
    u,s,vt=np.linalg.svd(matrix,full_matrices=False); rank=min(CFG['rank'],len(s)); scores=(u[:,:rank]*s[:rank])@vt[:rank,:]
    np.savez_compressed(ROOT/'models/personalized.npz',scores=scores,users=np.array(users),items=np.array(ids),seen=matrix.astype(bool))
    save(items[['item_id','popularity']].sort_values(['popularity','item_id'],ascending=[False,True]),'models/popularity.csv')
    write('models/parameters.json',{'personalized':{'algorithm':'truncated SVD of binary user-item matrix','rank':rank},'baseline':{'algorithm':'interaction count popularity'},'seed':CFG['seed']})

def evaluate():
    test=load('data/processed/test.csv'); items=load('data/features/item_features.csv'); d=np.load(ROOT/'models/personalized.npz'); users=d['users'].tolist(); ids=d['items'].tolist(); seen=d['seen']; scores=d['scores']; ui={u:i for i,u in enumerate(users)}
    pop=load('models/popularity.csv'); popscore=dict(zip(pop.item_id,pop.popularity)); categories=dict(zip(items.item_id,items.category)); k=CFG['k']; all_recs={}; summaries=[]
    # Leave seen items out of both relevance and candidate sets. Include cold users using popularity fallback.
    for model in ('popularity','personalized'):
        measures=[]; recommended=set(); pair_div=[]; excluded_no_novel=0
        for user,g in test.groupby('user_id',sort=True):
            known=set(np.array(ids)[seen[ui[user]]]) if user in ui else set(); relevant=set(g.item_id)-known
            if not relevant: excluded_no_novel+=1; continue
            candidates=[x for x in ids if x not in known]
            if model=='personalized' and user in ui:
                scoremap=dict(zip(ids,scores[ui[user]])); ranked=sorted(candidates,key=lambda x:(-scoremap[x],x))[:k]
            else: ranked=sorted(candidates,key=lambda x:(-popscore.get(x,0),x))[:k]
            hits=[int(x in relevant) for x in ranked]; dcg=sum(h/np.log2(i+2) for i,h in enumerate(hits)); ideal=sum(1/np.log2(i+2) for i in range(min(len(relevant),k)))
            measures.append((sum(hits)/k,sum(hits)/len(relevant),dcg/ideal)); recommended.update(ranked)
            pairs=[categories[a]!=categories[b] for i,a in enumerate(ranked) for b in ranked[i+1:]]; pair_div.append(float(np.mean(pairs)) if pairs else 0.)
            all_recs[f'{model}:{user}']=ranked
        if not measures: raise ValueError('No test users with novel relevant items')
        p,r,n=np.mean(measures,axis=0); summaries.append({'model':model,'precision_at_k':p,'recall_at_k':r,'ndcg_at_k':n,'coverage':len(recommended)/len(ids),'diversity':np.mean(pair_div),'evaluated_users':len(measures),'excluded_no_novel':excluded_no_novel})
    save(pd.DataFrame(summaries),'reports/model_comparison.csv'); write('reports/recommendations.json',all_recs)
    return summaries

def run():
    started=datetime.now(timezone.utc); run_id=started.strftime('%Y%m%dT%H%M%S%fZ'); state={'run_id':run_id,'started_utc':started.isoformat(),'config':CFG,'stages':[],'status':'running'}
    try:
        for name,fn in [('ingestion',ingest),('validation',validate),('preparation',prepare),('feature_transformation',features),('training',train),('evaluation',evaluate)]:
            t=time.monotonic(); result=fn(); state['stages'].append({'name':name,'status':'success','seconds':round(time.monotonic()-t,3)})
        snapshot=ROOT/'runs'/run_id; snapshot.mkdir(parents=True,exist_ok=True)
        for path in ('models/personalized.npz','models/popularity.csv','models/parameters.json','reports/model_comparison.csv','reports/validation_report.json','reports/split.json'):
            shutil.copy2(ROOT/path, snapshot/Path(path).name)
        state['artifact_directory']=str(snapshot.relative_to(ROOT)); state['metrics']=result; state['input_sha256']={n:hashlib.sha256((ROOT/f'data/raw/{n}.csv').read_bytes()).hexdigest() for n in ('users','items','interactions')}; state['status']='success'
    except Exception:
        state['status']='failed'; state['error']=traceback.format_exc(); raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat(); write(f'runs/{run_id}.json',state); write('reports/latest_run.json',state)
        print(json.dumps(state,indent=2))
if __name__=='__main__': run()
