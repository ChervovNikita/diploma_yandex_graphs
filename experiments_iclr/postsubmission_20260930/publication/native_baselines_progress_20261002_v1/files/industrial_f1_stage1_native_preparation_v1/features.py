"""Chronological adaptation of the pinned native graph builder."""
import os
from pathlib import Path

from common import sha256, write_json


class LocalGlove:
    def __init__(self, path, device, output_dir):
        # Offline-only is process configuration; root still owns confinement.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from sentence_transformers import SentenceTransformer
        path = Path(path).resolve(strict=True)
        files = {str(f.relative_to(path)): sha256(f) for f in sorted(path.rglob('*')) if f.is_file()}
        if not files or not (path / 'modules.json').is_file():
            raise ValueError('Expected a complete already-present local native GloVe asset')
        write_json(output_dir / 'GLOVE_ASSET.json', {
            'expected_native_model': 'sentence-transformers/average_word_embeddings_glove.6B.300d',
            'local_files_sha256': files,
            'note': 'Files recorded; semantic asset admission remains custodian responsibility'})
        self.model = SentenceTransformer(str(path), device=str(device), local_files_only=True)

    def __call__(self, sentences):
        import torch
        return torch.from_numpy(self.model.encode(sentences))


def chronological_graph(db, embedder_cfg, out):
    import numpy as np
    import pandas as pd
    import torch
    from torch_frame import stype
    from torch_frame.data import Dataset
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import sort_edge_index
    from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey, to_unix_time
    fit_db = db.upto(pd.Timestamp('2005-01-01'))
    # Exactly native proposal, with fixed preparation RNG and chronological input.
    stypes = get_stype_proposal(fit_db)
    write_json(out / 'STYPES.json', {n: {c: str(s) for c, s in cs.items()} for n, cs in stypes.items()})
    data, stats, fitted, audit = HeteroData(), {}, {}, {}
    for name, table in db.table_dict.items():
        full = table.df.copy()
        fit = fit_db.table_dict[name].df.copy()
        cs = stypes[name].copy()
        remove_pkey_fkey(cs, table)
        if len(fit) == 0:
            raise ValueError(f'No chronological fit rows for table {name}')
        if not cs:
            cs = {'__const__': stype.numerical}
            full['__const__'], fit['__const__'] = np.ones(len(full)), np.ones(len(fit))
        ds = Dataset(df=fit, col_to_stype=cs,
                     col_to_text_embedder_cfg=embedder_cfg).materialize()
        # Use the fitted converter; never materialize a separate full-table Dataset.
        data[name].tf = ds.convert_to_tensor_frame(full)
        data[name].num_nodes = len(full)
        stats[name], fitted[name] = ds.col_stats, ds
        if table.time_col:
            data[name].time = torch.from_numpy(to_unix_time(table.df[table.time_col]))
        audit[name] = {'fit_rows': len(fit), 'public_rows': len(full),
                       'fit_max_time': str(fit[table.time_col].max()) if table.time_col else None,
                       'untimestamped_native_metadata': table.time_col is None,
                       'feature_columns': list(cs)}
        for col, target in table.fkey_col_to_pkey_table.items():
            keys = table.df[col]
            valid = ~keys.isna()
            src = torch.arange(len(keys))[torch.from_numpy(valid.to_numpy())]
            dst = torch.from_numpy(keys[valid].astype(int).to_numpy())
            data[name, f'f2p_{col}', target].edge_index = sort_edge_index(torch.stack([src, dst]))
            data[target, f'rev_f2p_{col}', name].edge_index = sort_edge_index(torch.stack([dst, src]))
    data.validate(raise_on_error=True)
    torch.save({'graph': data, 'col_stats': stats, 'fit_datasets': fitted}, out / 'FROZEN_FEATURES.pt')
    write_json(out / 'FEATURE_FIT_AUDIT.json', audit)
    return data, stats, stypes
