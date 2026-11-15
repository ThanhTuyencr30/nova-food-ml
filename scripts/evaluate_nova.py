"""Evaluate handed-off NOVA predictions without retraining or changing source files.
Run: python evaluate_nova.py --project PATH --output PATH
Requires: pandas, numpy, scikit-learn, matplotlib.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

LABELS = [1, 2, 3, 4]

def normalized(s):
    # Same lowercasing, Unicode accent stripping and token pattern as the supplied TF-IDF.
    return ' '.join(re.findall(r'(?u)\b\w\w+\b', strip_accents_unicode(str(s).lower())))

def evaluate(project, output):
    output.mkdir(parents=True, exist_ok=True)
    files = ['data/processed/off_nova.csv', 'data/processed/vietnam.csv',
             'models/model_meta.json', 'models/nova_pipeline.joblib',
             'reports/model_comparison.csv', 'reports/test_predictions.csv',
             'reports/vietnam_predictions.csv']
    manifest = []
    for f in files:
        p = project / f
        manifest.append({'file': f, 'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    d = pd.read_csv(project/files[0]); v = pd.read_csv(project/files[1])
    meta = json.loads((project/'models/model_meta.json').read_text(encoding='utf-8'))
    train, test = train_test_split(d, test_size=meta['test_size'],
                                  random_state=meta['random_state'], stratify=d.nova_group)
    train = train.copy(); train['_norm'] = train.ingredients_text.map(normalized)
    exact_seen = set(train.ingredients_text); norm_seen = set(train._norm)
    train_exact = train.groupby('ingredients_text').nova_group.agg(lambda x: ','.join(map(str,sorted(set(x)))))
    train_norm = train.groupby('_norm').nova_group.agg(lambda x: ','.join(map(str,sorted(set(x)))))
    summary, per_class, overlap, confusions = [], [], [], []
    annotations = {}
    for name, source, filename in [('test', test, 'test_predictions.csv'),
                                    ('vietnam', v, 'vietnam_predictions.csv')]:
        source = source.reset_index().rename(columns={'index':'processed_row_0based'})
        pred = pd.read_csv(project/'reports'/filename)
        for col in ['ingredients_text','product_name']:
            assert source[col].fillna('').equals(pred[col].fillna('')), f'{name}: {col} does not match split'
        assert source.nova_group.equals(pred.y_true), f'{name}: labels do not match'
        assert pred.y_true.isin(LABELS).all() and pred.y_pred.isin(LABELS).all()
        x = pred.copy(); x['prediction_row_0based'] = np.arange(len(x))
        x['processed_row_0based'] = source.processed_row_0based
        x['normalized_text'] = x.ingredients_text.map(normalized)
        x['seen_exact_in_train'] = x.ingredients_text.isin(exact_seen)
        x['seen_normalized_in_train'] = x.normalized_text.isin(norm_seen)
        x['train_labels_exact'] = x.ingredients_text.map(train_exact).fillna('')
        x['train_labels_normalized'] = x.normalized_text.map(train_norm).fillna('')
        x['wrong'] = x.y_true.ne(x.y_pred)
        annotations[name] = x
        x.to_csv(output/f'{name}_predictions_annotated.csv', index=False, encoding='utf-8-sig')
        x[x.wrong].to_csv(output/f'{name}_errors.csv', index=False, encoding='utf-8-sig')
        slices = {'all': np.ones(len(x), dtype=bool), 'seen_exact':x.seen_exact_in_train,
                  'unseen_exact':~x.seen_exact_in_train,
                  'seen_normalized':x.seen_normalized_in_train,
                  'unseen_normalized':~x.seen_normalized_in_train}
        for slice_name, mask in slices.items():
            s=x.loc[mask]
            if len(s)==0: continue
            cm = confusion_matrix(s.y_true,s.y_pred,labels=LABELS)
            precision, recall, f1, support = precision_recall_fscore_support(
                s.y_true,s.y_pred,labels=LABELS,zero_division=0)
            summary.append({'dataset':name,'slice':slice_name,'n':len(s),'errors':int(s.wrong.sum()),
                            'macro_f1':float(f1.mean()),'accuracy':accuracy_score(s.y_true,s.y_pred),
                            'classes_present':int((support>0).sum())})
            for i,label in enumerate(LABELS):
                per_class.append({'dataset':name,'slice':slice_name,'nova':label,'support':int(support[i]),
                                  'predicted':int(cm[:,i].sum()),'correct':int(cm[i,i]),
                                  'precision':precision[i],'recall':recall[i],'f1':f1[i]})
            if slice_name=='all':
                assert abs(f1.mean()-meta[name if name=='vietnam' else 'test']['macro_f1'])<1e-10
        for kind, flag in [('exact','seen_exact_in_train'),('normalized','seen_normalized_in_train')]:
            s=x[x[flag]]
            overlap.append({'dataset':name,'matching':kind,'rows_seen':len(s),'total':len(x),
                            'fraction':len(s)/len(x),
                            'distinct_seen_texts':s['ingredients_text' if kind=='exact' else 'normalized_text'].nunique()})
        for scope in ['all','unseen_normalized']:
            s=x if scope=='all' else x[~x.seen_normalized_in_train]
            cm=confusion_matrix(s.y_true,s.y_pred,labels=LABELS)
            denom=cm.sum(axis=1,keepdims=True)
            cmnorm=np.divide(cm,denom,out=np.zeros_like(cm,dtype=float),where=denom!=0)
            for mode, arr in [('counts',cm),('row_normalized',cmnorm)]:
                pd.DataFrame(arr,index=[f'true_{i}' for i in LABELS],columns=[f'pred_{i}' for i in LABELS]).to_csv(output/f'cm_{name}_{scope}_{mode}.csv',encoding='utf-8-sig')
            fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
            for ax,arr,title in zip(axes,[cm,cmnorm],['Counts','Row-normalized (%)']):
                ax.imshow(arr,cmap='Blues',vmin=0,vmax=None if title=='Counts' else 1)
                for i in range(4):
                    for j in range(4):
                        txt=str(cm[i,j]) if title=='Counts' else f'{arr[i,j]*100:.1f}%'
                        ax.text(j,i,txt,ha='center',va='center',color='white' if arr[i,j]>arr.max()/2 else '#17324d',fontsize=11)
                ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,
                       xlabel='Predicted NOVA',ylabel='True NOVA',title=title)
            fig.suptitle(f'{name.upper()} | {scope} | n = {len(s):,}',fontsize=14)
            fig.savefig(output/f'confusion_{name}_{scope}.png',dpi=180);plt.close(fig)
            if scope=='all':
                for i in range(4):
                    for j in range(4):
                        if i!=j: confusions.append({'dataset':name,'y_true':i+1,'y_pred':j+1,
                            'n':int(cm[i,j]),'fraction_of_true_class':float(cmnorm[i,j])})
    pd.DataFrame(summary).to_csv(output/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(output/'metrics_per_class.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(overlap).to_csv(output/'overlap_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(confusions).sort_values(['dataset','n'],ascending=[True,False]).to_csv(output/'confusion_pairs.csv',index=False,encoding='utf-8-sig')
    result={'summary':summary,'per_class':per_class,'overlap':overlap,
            'confusions':confusions,'meta':meta,
            'train_n':len(train),'processed_n':len(d),
            'train_exact_conflicting_texts':int((train.groupby('ingredients_text').nova_group.nunique()>1).sum()),
            'train_normalized_conflicting_texts':int((train.groupby('_norm').nova_group.nunique()>1).sum())}
    (output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();r=evaluate(args.project,args.output)
    print(json.dumps({'summary':r['summary'],'overlap':r['overlap'],'conflicts':r['train_normalized_conflicting_texts']},ensure_ascii=False,indent=2))
