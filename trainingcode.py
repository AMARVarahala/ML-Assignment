"""Reproducible polynomial-only regression solution for IMT2024085.

Usage:
    python train_predict.py
    python train_predict.py --data-dir ./data --output-dir ./outputs

Requirements: numpy, scipy, scikit-learn, matplotlib, joblib.
All selection and performance estimates use only supplied training targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = 'IMT2024085'
FOLDS = KFold(n_splits=5, shuffle=True, random_state=812)


def read_numeric_csv(path: Path, expected_columns):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f'{path.name}: expected {expected_columns}, got {reader.fieldnames}')
        data = np.asarray([[float(row[c]) for c in expected_columns] for row in reader], dtype=float)
    if data.ndim != 2 or data.shape[0] < 1 or not np.all(np.isfinite(data)):
        raise ValueError(f'Invalid numeric data: {path}')
    return data


def write_predictions(path: Path, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as h:
        writer = csv.writer(h)
        writer.writerow(['y'])
        writer.writerows([[f'{float(z):.15g}'] for z in y_pred])


class PolynomialEnsemble:
    """A finite linear combination of polynomial-regression models."""
    def __init__(self, variation):
        if variation not in (1, 2):
            raise ValueError('Expected variation 1 or 2')
        self.variation = variation

    def fit(self, X, y):
        self.models_ = []
        if self.variation == 1:
            # Both members are degree-five polynomials in six input variables.
            power = PolynomialFeatures(degree=5, include_bias=False)
            Z = power.fit_transform(X)
            scale = StandardScaler().fit(Z)
            A = scale.transform(Z)
            sparse = Lasso(alpha=0.01, max_iter=4000, tol=1e-4).fit(A, y)
            selection = Lasso(alpha=0.02, max_iter=4000, tol=1e-4).fit(A, y)
            selected = np.abs(selection.coef_) > 1e-6
            if not np.any(selected):
                raise RuntimeError('No polynomial terms selected for Phase 1')
            refit = LinearRegression().fit(A[:, selected], y)
            self.models_.append({'weight': 0.5, 'power': power, 'scale': scale,
                                 'estimator': sparse, 'selected': None})
            self.models_.append({'weight': 0.5, 'power': power, 'scale': scale,
                                 'estimator': refit, 'selected': selected})
            self.selected_terms_ = int(selected.sum())
        else:
            # The mixture is still a polynomial, of maximum total degree 12.
            self.selected_terms_ = None
            for degree, alpha, weight in [(10, 1.0, 0.25),
                                          (11, 1.0, 0.25),
                                          (12, 3.0, 0.50)]:
                power = PolynomialFeatures(degree=degree, include_bias=False)
                Z = power.fit_transform(X)
                scale = StandardScaler().fit(Z)
                estimator = Ridge(alpha=alpha).fit(scale.transform(Z), y)
                self.models_.append({'weight': weight, 'power': power,
                                     'scale': scale, 'estimator': estimator,
                                     'selected': None})
        self.training_rows_ = len(X)
        return self

    def predict(self, X):
        prediction = np.zeros(len(X), dtype=float)
        for m in self.models_:
            A = m['scale'].transform(m['power'].transform(X))
            if m['selected'] is not None:
                A = A[:, m['selected']]
            prediction += m['weight'] * m['estimator'].predict(A)
        return prediction


# Keep serialized model classes importable even when this script runs as __main__.
sys.modules.setdefault("train_predict", sys.modules[__name__])
PolynomialEnsemble.__module__ = "train_predict"


def evaluate_cv(X, y, variation):
    folds = []
    oof = np.empty(len(y), dtype=float)
    for k, (train_idx, val_idx) in enumerate(FOLDS.split(X), 1):
        model = PolynomialEnsemble(variation).fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        oof[val_idx] = pred
        folds.append({'fold': k, 'R2': float(r2_score(y[val_idx], pred)),
                      'MSE': float(mean_squared_error(y[val_idx], pred))})
    return folds, oof


def degree_comparison(X, y, variation):
    """One-model baselines for the validation plots; no test targets used."""
    if variation == 1:
        config = [(2, 'ridge', 1.), (3, 'ridge', 10.), (4, 'lasso', .01),
                  (5, 'lasso', .01), (6, 'lasso', .01), (7, 'lasso', .01)]
    else:
        config = [(2, 1.), (4, 1.), (6, .03), (8, .03), (10, 1.),
                  (11, 1.), (12, 3.), (14, 3.), (16, 3.), (20, 10.)]
    out = []
    for setting in config:
        degree = setting[0]
        Z = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
        scores = []
        for t_idx, v_idx in FOLDS.split(X):
            scale = StandardScaler().fit(Z[t_idx])
            tr = scale.transform(Z[t_idx]); va = scale.transform(Z[v_idx])
            if variation == 1:
                method, alpha = setting[1:]
                reg = Lasso(alpha=alpha, max_iter=4000, tol=1e-4) if method=='lasso' else Ridge(alpha=alpha)
            else:
                reg = Ridge(alpha=setting[1])
            reg.fit(tr, y[t_idx])
            scores.append(r2_score(y[v_idx], reg.predict(va)))
        out.append({'degree':degree, 'n_terms':Z.shape[1],
                    'mean_R2':float(np.mean(scores)), 'std_R2':float(np.std(scores)),
                    'model':(setting[1] if variation==1 else 'ridge'),
                    'alpha':float(setting[2] if variation==1 else setting[1])})
    return out


def write_degree_plot(summaries, report_dir):
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=160)
    for ax, v in zip(axs, (1, 2)):
        rows = summaries[str(v)]['degree_comparison']
        d = [r['degree'] for r in rows]; yy = [r['mean_R2'] for r in rows]
        ax.plot(d, yy, marker='o', lw=2, markersize=4, color='#17679A')
        ax.set_title(f'Phase {v}: 5-fold validation', fontweight='bold', fontsize=11)
        ax.set_xlabel('Maximum total polynomial degree');ax.set_ylabel('Mean validation R²')
        ax.grid(alpha=.23);ax.set_xticks(d)
        ax.set_ylim(max(0,min(yy)-.08),min(1.02,max(yy)+.025))
    fig.tight_layout()
    fig.savefig(report_dir / 'degree_comparison.png', dpi=190, bbox_inches='tight')
    plt.close(fig)


def write_pred_chart(summaries, report_dir):
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)
    for ax, v in zip(axs, (1, 2)):
        y = np.asarray(summaries[str(v)]['y_true'])
        ypred = np.asarray(summaries[str(v)]['oof_pred'])
        ax.scatter(y, ypred, alpha=.3, s=9, color='#257DA2', edgecolors='none')
        low=min(y.min(),ypred.min());high=max(y.max(),ypred.max())
        ax.plot([low,high],[low,high],color='#D46A40',lw=1.5,ls='--')
        ax.set_xlabel('Observed training target (held-out fold)')
        ax.set_ylabel('Out-of-fold prediction')
        ax.set_title(f'Phase {v}: out-of-fold predictions',fontsize=11,fontweight='bold')
        ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(report_dir / 'out_of_fold_predictions.png',dpi=190,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--report-dir',type=Path,default=Path(__file__).resolve().parent/'reports')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.report_dir.mkdir(parents=True,exist_ok=True)
    report={}
    sample=read_numeric_csv(args.data_dir/'sample_submission.csv',['y'])
    for v in (1,2):
        cols=[f'x{i}' for i in range(1,7 if v==1 else 4)]
        train=read_numeric_csv(args.data_dir/f'{ROLL}_train_var{v}.csv',cols+['y'])
        test=read_numeric_csv(args.data_dir/f'{ROLL}_test_var{v}.csv',cols)
        if len(test)!=len(sample):
            raise ValueError('Test row count disagrees with sample submission')
        X,y=train[:,:-1],train[:,-1]
        folds,oof=evaluate_cv(X,y,v)
        degree=degree_comparison(X,y,v)
        model=PolynomialEnsemble(v).fit(X,y)
        pred=model.predict(test)
        if not np.all(np.isfinite(pred)):
            raise RuntimeError(f'Phase {v} produced nonfinite predictions')
        csv_path=args.output_dir/f'{ROLL}-pred_var{v}.csv'
        write_predictions(csv_path,pred)
        joblib.dump(model,args.output_dir/f'phase{v}_polynomial_model.joblib')
        report[str(v)]={'n_train':len(X),'n_test':len(test),'n_features':X.shape[1],
                        'mean_cv_r2':float(np.mean([f['R2'] for f in folds])),
                        'std_cv_r2':float(np.std([f['R2'] for f in folds])),
                        'mean_cv_mse':float(np.mean([f['MSE'] for f in folds])),
                        'fold_results':folds, 'degree_comparison':degree,
                        'full_fit_train_r2':float(r2_score(y,model.predict(X))),
                        'selected_terms_phase1':model.selected_terms_,
                        'pred_range':[float(np.min(pred)),float(np.max(pred))],
                        'target_range':[float(np.min(y)),float(np.max(y))],
                        'y_true':y.tolist(),'oof_pred':oof.tolist()}
        print(f'Phase {v} -> {csv_path.name}: CV R2={report[str(v)]["mean_cv_r2"]:.6f} '
              f'MSE={report[str(v)]["mean_cv_mse"]:.6f}, test rows={len(pred)}')
    write_degree_plot(report,args.report_dir)
    write_pred_chart(report,args.report_dir)
    for v in (1,2):
        del report[str(v)]['y_true'];del report[str(v)]['oof_pred']
    with (args.report_dir/'validation_summary.json').open('w') as out:
        json.dump(report,out,indent=2)
    print('Prediction format checked against sample submission; all models polynomial-only.')


if __name__=='__main__':
    main()
