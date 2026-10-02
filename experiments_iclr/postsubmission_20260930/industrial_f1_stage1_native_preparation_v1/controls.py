"""Fixed median/history and two native ten-trial LightGBM controls."""
import json
import time

from common import PACKET, metrics, write_json, write_predictions


def run_gbdt(name, dfs, col_to_stype, embedder_cfg, rows, out):
    import lightgbm
    import optuna
    import torch
    from torch_frame import stype
    from torch_frame.data import Dataset
    from torch_frame.gbdt import LightGBM
    from torch_frame.typing import Metric

    class RecordedLightGBM(LightGBM):
        def objective(self, trial, *args, **kwargs):
            started = time.perf_counter()
            score = None
            try:
                # Exact native objective/search domain and early stopping.
                score = super().objective(trial, *args, **kwargs)
                return score
            finally:
                record = {'trial': trial.number, 'value': score,
                          'seconds': time.perf_counter() - started,
                          'suggested_params': trial.params,
                          'native_booster_params': getattr(self, 'params', {}),
                          'booster_seed': 'LightGBM native default 0 (not overridden)'}
                write_json(out / f'{name}_trial{trial.number:02d}.json', record)

        def _tune(self, tf_train, tf_val, num_trials, num_boost_round=2000):
            # Native _tune with two observability/repeatability changes:
            # retain study and use seeded TPE. Domain/objective/refit are unchanged.
            self.study = optuna.create_study(
                direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                storage=f'sqlite:///{out / (name + "_trials.sqlite3")}',
                study_name=name)
            train_x, train_y, cats = self._to_lightgbm_input(tf_train)
            val_x, val_y, _ = self._to_lightgbm_input(tf_val)
            train_data = lightgbm.Dataset(train_x, label=train_y, free_raw_data=False)
            eval_data = lightgbm.Dataset(val_x, label=val_y, free_raw_data=False)
            self.effective_input = {'columns': len(train_x.columns),
                                    'categorical_indices': cats,
                                    'frame_feature_columns': {str(s): cs for s, cs in tf_train.col_names_dict.items()},
                                    'timestamp_tensors_ignored_by_native_converter': True}
            tune_start = time.perf_counter()
            self.study.optimize(lambda trial: self.objective(
                trial, train_data, eval_data, cats, num_boost_round), n_trials=num_trials)
            self.search_seconds = time.perf_counter() - tune_start
            self.params.update(self.study.best_params)
            refit_start = time.perf_counter()
            self.model = lightgbm.train(
                self.params, train_data, num_boost_round=num_boost_round,
                categorical_feature=cats, valid_sets=[eval_data],
                callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                           lightgbm.log_evaluation(period=2000)])
            self.final_refit_seconds = time.perf_counter() - refit_start

    started = time.perf_counter()
    cs = dict(col_to_stype)
    cs['position'] = stype.numerical
    ds = Dataset(df=dfs['train'], col_to_stype=cs, target_col='position',
                 col_to_text_embedder_cfg=embedder_cfg).materialize()
    tf_val = ds.convert_to_tensor_frame(dfs['val'])
    materialize_seconds = time.perf_counter() - started
    torch.save(ds, out / f'{name}_FROZEN_TRAIN_CONVERTER.pt')
    model = RecordedLightGBM(task_type=ds.task_type, metric=Metric.MAE)
    model.tune(tf_train=ds.tensor_frame, tf_val=tf_val, num_trials=10)
    if len(model.study.trials) != 10 or any(t.state != optuna.trial.TrialState.COMPLETE for t in model.study.trials):
        raise RuntimeError('Every declared GBDT view requires exactly ten completed trials')
    pred = model.predict(tf_test=tf_val).numpy()
    score = metrics(rows['val'].df.position, pred)
    write_predictions(out / f'{name}_val.csv', rows['val'], pred)
    model.save(str(out / f'{name}_best.txt'))
    report = {'validation': score, 'trials': [{
        'number': t.number, 'state': t.state.name, 'value': t.value, 'params': t.params,
        'duration_seconds': t.duration.total_seconds() if t.duration else None}
        for t in model.study.trials], 'best_trial': model.study.best_trial.number,
        'best_params': model.study.best_params, 'final_booster_params': model.model.params,
        'final_best_iteration': model.model.best_iteration,
        'optuna_sampler': 'TPESampler', 'optuna_seed': 42,
        'booster_seed': 'native default 0', 'sample_size': 0, 'use_ar_label': False,
        'materialization_seconds': materialize_seconds, 'search_seconds': model.search_seconds,
        'final_refit_seconds': model.final_refit_seconds,
        'total_seconds': time.perf_counter() - started, 'effective_input': model.effective_input}
    write_json(out / f'{name}_summary.json', report)
    return report


def history_frame(db, table):
    import duckdb
    import numpy as np
    conn = duckdb.connect(':memory:')
    try:
        for name, t in db.table_dict.items():
            conn.register(name, t.df)
        conn.register('forecast_rows', table.df)
        df = conn.execute((PACKET / 'history_features.sql').read_text()).df()
    finally:
        conn.close()
    cs = json.loads((PACKET / 'ENGINEERED_STYPES.json').read_text())
    if set(df.columns) != set(cs) | {'row_id', 'row_hash'} or len(df) != len(table):
        raise RuntimeError('History view has unexpected feature columns or row multiplicity')
    if not np.array_equal(df.row_id.to_numpy(), table.df.row_id.to_numpy()):
        raise RuntimeError('History view lost original forecast order')
    for col in ['row_hash', 'driverId', 'date', 'position']:
        if not df[col].reset_index(drop=True).equals(table.df[col].reset_index(drop=True)):
            # DuckDB may change datetime precision; numeric/key/date values must agree.
            if not np.array_equal(df[col].to_numpy(), table.df[col].to_numpy()):
                raise RuntimeError(f'History view key/target mismatch: {col}')
    return df


def run_controls(db, rows, graph_stypes, embedder_cfg, out):
    import numpy as np
    import pandas as pd
    from torch_frame import stype
    from relbench.modeling.utils import remove_pkey_fkey
    reports = {}
    median = float(np.median(rows['train'].df.position.to_numpy()))
    started = time.perf_counter()
    pred = np.full(len(rows['val']), median)
    reports['median'] = {'training_global_median': median,
                         'validation': metrics(rows['val'].df.position, pred),
                         'seconds': time.perf_counter() - started}
    write_predictions(out / 'median_val.csv', rows['val'], pred)
    started = time.perf_counter()
    results = db.table_dict['results'].df
    pred, fallback = [], 0
    for r in rows['val'].df.itertuples():
        past = results[(results.driverId == r.driverId) & (results.date <= r.date) &
                       (results.date > r.date - pd.Timedelta(days=365))]
        values = pd.to_numeric(past.positionOrder, errors='coerce').to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            pred.append(float(values.mean()))
        else:
            pred.append(median)
            fallback += 1
    pred = np.asarray(pred)
    reports['driver_history'] = {
        'rule': 'Mean results.positionOrder for the same driver in (t-365 days,t]; training global median fallback',
        'fallback_rows': fallback, 'validation': metrics(rows['val'].df.position, pred),
        'seconds': time.perf_counter() - started}
    write_predictions(out / 'driver_history_val.csv', rows['val'], pred)
    drivers = db.table_dict['drivers']
    cs = graph_stypes['drivers'].copy()
    remove_pkey_fkey(cs, drivers)
    raw = {}
    for split in ['train', 'val']:
        raw[split] = rows[split].df.merge(drivers.df, how='left', on='driverId', validate='many_to_one', sort=False)
        if raw[split].row_id.tolist() != rows[split].df.row_id.tolist():
            raise RuntimeError('Native raw merge changed forecast row order')
    reports['raw_gbdt'] = run_gbdt('raw_gbdt', raw, cs, embedder_cfg, rows, out)
    feature_start = time.perf_counter()
    history = {split: history_frame(db, rows[split]) for split in ['train', 'val']}
    feature_seconds = time.perf_counter() - feature_start
    cs = {col: stype(kind) for col, kind in json.loads((PACKET / 'ENGINEERED_STYPES.json').read_text()).items()}
    reports['history_gbdt'] = run_gbdt('history_gbdt', history, cs, None, rows, out)
    reports['history_gbdt']['sql_feature_seconds'] = feature_seconds
    write_json(out / 'CONTROLS_SUMMARY.json', reports)
    return reports
