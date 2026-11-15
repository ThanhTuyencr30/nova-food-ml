"""NOVA keyword, input-shift and error-review preparation. No training.
Run with a trusted handed-off model: python analyze_nova_details.py --project PATH --output PATH
"""
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def analyze(project, out):
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(project/'data/processed/off_nova.csv')
    vn=pd.read_csv(project/'data/processed/vietnam.csv')
    meta=json.loads((project/'models/model_meta.json').read_text())
    tr,te=train_test_split(d,test_size=meta['test_size'],random_state=meta['random_state'],stratify=d.nova_group)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always');model=joblib.load(project/'models/nova_pipeline.joblib')
    vec=model.named_steps['features'].named_transformers_['tfidf']
    clf=model.named_steps['classifier'];terms=vec.get_feature_names_out()
    assert list(clf.classes_)==[1,2,3,4]
    assert clf.coef_.shape==(4,len(terms)) and meta['features']=='text_only'
    tok=vec.build_tokenizer(); prep=vec.build_preprocessor()
    unigram_vocab={t for t in terms if ' ' not in t}
    Xtrain=vec.transform(tr.ingredients_text)
    kw=[]
    for ci,lab in enumerate(clf.classes_):
        mask=tr.nova_group.to_numpy()==lab
        for rank,i in enumerate(np.argsort(clf.coef_[ci])[::-1][:20],1):
            counts=[int(Xtrain[tr.nova_group.to_numpy()==l,i].getnnz()) for l in clf.classes_]
            kw.append({'nova':int(lab),'rank':rank,'term':terms[i],'coefficient':float(clf.coef_[ci,i]),
                'train_documents_any_class':sum(counts),'train_documents_target_class':counts[ci],
                **{f'docs_nova_{l}':counts[j] for j,l in enumerate(clf.classes_)}})
    k=pd.DataFrame(kw);k.to_csv(out/'top_terms_by_nova.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for lab,ax in zip(clf.classes_,axes.flat):
        sub=k[k.nova==lab].head(10).iloc[::-1]
        ax.barh(sub.term,sub.coefficient,color=['#3977a3','#4b908d','#d19751','#8b678e'][lab-1])
        ax.set(title=f'NOVA {lab} | Top 10 positive coefficients',xlabel='LinearSVC coefficient')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'keywords_nova.png',dpi=160);plt.close(fig)
    stats=[];nut=[];classes=[];details={}
    for name,src in [('train',tr),('test',te),('vietnam',vn)]:
        src=src.reset_index(drop=True).copy()
        tokens=src.ingredients_text.map(lambda x:tok(prep(str(x))))
        src['n_tokens']=tokens.map(len)
        src['n_chars']=src.ingredients_text.str.len()
        src['n_oov']=tokens.map(lambda ts:sum(t not in unigram_vocab for t in ts))
        src['oov_ratio']=np.divide(src.n_oov,src.n_tokens,out=np.zeros(len(src)),where=src.n_tokens.to_numpy()!=0)
        src['has_vi_chars']=src.ingredients_text.str.contains('[ăđơư\u1ea0-\u1ef9]',case=False,regex=True)
        X=vec.transform(src.ingredients_text)
        src['zero_vector']=np.asarray(X.getnnz(axis=1)==0)
        stats.append({'dataset':name,'n':len(src),'median_chars':float(src.n_chars.median()),
            'median_tokens':float(src.n_tokens.median()),'p90_tokens':float(src.n_tokens.quantile(.9)),
            'mean_oov_fraction_per_product':float(src.oov_ratio.mean()),
            'token_weighted_oov_fraction':float(src.n_oov.sum()/src.n_tokens.sum()),
            'zero_vectors':int(src.zero_vector.sum()),'vi_char_rows':int(src.has_vi_chars.sum()),
            'vi_char_fraction':float(src.has_vi_chars.mean())})
        for l in [1,2,3,4]:
            n=int((src.nova_group==l).sum());classes.append({'dataset':name,'nova':l,'n':n,'fraction':n/len(src)})
        for col in ['energy-kcal_100g','fat_100g','saturated-fat_100g','sugars_100g','salt_100g','proteins_100g','fiber_100g']:
            exists=col in src
            s=pd.to_numeric(src[col],errors='coerce') if exists else pd.Series(np.nan,index=src.index)
            nut.append({'dataset':name,'column':col,'column_exists':exists,'missing_n':int(s.isna().sum()),
                        'missing_fraction':float(s.isna().mean()),'median_available':float(s.median()) if s.notna().any() else None})
        details[name]=src
    pd.DataFrame(stats).to_csv(out/'input_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(nut).to_csv(out/'nutrition_missingness.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(classes).to_csv(out/'class_distribution.csv',index=False,encoding='utf-8-sig')
    errors=[]
    prediction_check={}
    for name in ['test','vietnam']:
        pred=pd.read_csv(project/'reports'/f'{name}_predictions.csv')
        src=details[name]
        assert src.ingredients_text.equals(pred.ingredients_text) and src.nova_group.equals(pred.y_true)
        recomputed=model.predict(src[['ingredients_text']])
        prediction_check[name]=int(np.sum(recomputed!=pred.y_pred))
        assert prediction_check[name]==0
        X=vec.transform(src.ingredients_text)
        all_scores=model.decision_function(src[['ingredients_text']])
        for i in pred.index[pred.y_true.ne(pred.y_pred)]:
            x=X[i];p=int(pred.at[i,'y_pred']);t=int(pred.at[i,'y_true'])
            contrib=x.data*(clf.coef_[p-1,x.indices]-clf.coef_[t-1,x.indices])
            ids=np.argsort(contrib)[::-1]
            positives=[f'{terms[x.indices[j]]}: {contrib[j]:.4f}' for j in ids if contrib[j]>0][:5]
            negatives=[f'{terms[x.indices[j]]}: {contrib[j]:.4f}' for j in np.argsort(contrib) if contrib[j]<0][:5]
            scores=all_scores[i]
            intercept_diff=float(clf.intercept_[p-1]-clf.intercept_[t-1])
            assert abs(contrib.sum()+intercept_diff-(scores[p-1]-scores[t-1]))<1e-8
            errors.append({'dataset':name,'prediction_row_0based':int(i),'product_name':pred.at[i,'product_name'],
                'ingredients_text':pred.at[i,'ingredients_text'],'y_true':t,'y_pred':p,
                'n_tokens':int(src.at[i,'n_tokens']),'oov_ratio':float(src.at[i,'oov_ratio']),
                'zero_vector':bool(src.at[i,'zero_vector']),'has_vi_chars':bool(src.at[i,'has_vi_chars']),
                'supports_prediction_vs_true':'; '.join(positives),
                'supports_true_vs_prediction':'; '.join(negatives),
                'token_contribution_sum':float(contrib.sum()),
                'intercept_difference':intercept_diff,
                'pred_minus_true_margin':float(scores[p-1]-scores[t-1])})
    allerr=pd.DataFrame(errors)
    allerr.to_csv(out/'errors_with_model_evidence.csv',index=False,encoding='utf-8-sig')
    selected=[allerr[allerr.dataset=='vietnam']]
    for _,s in allerr[allerr.dataset=='test'].groupby(['y_true','y_pred'],sort=True):
        selected.append(s.sample(min(3,len(s)),random_state=42))
    sample=pd.concat(selected,ignore_index=True)
    sample.to_csv(out/'error_review_sample.csv',index=False,encoding='utf-8-sig')
    for name in ['test','vietnam']:
        src=details[name];p=pd.read_csv(project/'reports'/f'{name}_predictions.csv');wrong=p.y_true.ne(p.y_pred)
        for correct,mask in [('correct',~wrong),('wrong',wrong)]:
            sub=src[mask]
            stats.append({'dataset':f'{name}_{correct}','n':len(sub),'median_chars':float(sub.n_chars.median()),
                          'median_tokens':float(sub.n_tokens.median()),'mean_oov_fraction_per_product':float(sub.oov_ratio.mean()),
                          'vi_char_rows':int(sub.has_vi_chars.sum())})
    summary={'input_stats':stats,'nutrition':nut,'class_distribution':classes,
             'load_warnings':[str(w.message) for w in warns],'prediction_mismatches':prediction_check,
             'review_n':len(sample),'review_test_n':int((sample.dataset=='test').sum()),
             'review_vietnam_n':int((sample.dataset=='vietnam').sum())}
    (out/'details_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();analyze(a.project,a.output)
