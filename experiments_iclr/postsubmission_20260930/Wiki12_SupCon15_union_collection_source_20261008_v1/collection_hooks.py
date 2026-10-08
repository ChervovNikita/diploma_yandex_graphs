"""Identity/restore hooks for the unchanged Wiki12 collector's fifteen states."""
import hashlib
import math

WIKI = ('plain', 'alignment_only', 'residual_only', 'combined')


def require(value, message):
    if not value:
        raise ValueError(message)


class Adapter:
    def __init__(self, original, supcon, recompute):
        self.original, self.supcon, self.recompute = original, supcon, recompute

    def identity(self, condition):
        if condition in WIKI:
            return self.original.identity(condition)
        require(condition == 'supcon_eq2', 'Exact canonical candidate')
        specification = self.supcon.identity(condition, self.recompute)
        # Only the collector API needs control_id; the original saved config does not contain it.
        return dict(specification, control_id=specification['method_identity'])

    def configured_recipe(self, public, condition):
        if condition in WIKI:
            return self.original.configured_recipe(public, condition)
        return self.supcon.recipe(public, self.supcon.identity(condition, self.recompute))


def restore_factory(original_restore):
    def restore(torch, model, saved, record, config, specification, pins, public):
        if record['condition'] in WIKI:
            return original_restore(torch, model, saved, record, config, specification, pins, public)
        expected = {key: value for key, value in specification.items() if key != 'control_id'}
        require(isinstance(saved, dict) and saved.get('config') == config
                and saved.get('public_loss_comparison') == expected, 'Original canonical selected config/loss identity')
        run = saved['run']
        require(record['condition'] == 'supcon_eq2' and run['public_loss_comparison'] == expected
                and run['task'] == 'wikics' and run['seed'] == record['seed']
                and run['arm'] == run['method_identity'] == record['method_identity']
                and run['TEST_scoring'] is False and run['part_of_registered_author_Wiki12'] is False
                and expected['public_wrapper_sha256'] == pins['supcon_wrapper_sha256']
                and expected['loss_module_sha256'] == pins['supcon_loss_sha256']
                and expected['recompute_sha256'] == pins['supcon_recompute_sha256'],
                'Distinct original SupCon method/source/seed; no Wiki12 relabeling')
        core = {name: hashlib.sha256((public.ROOT / 'core' / name).read_bytes()).hexdigest()
                for name in ('factors.py', 'models.py', 'objectives.py', 'selection.py')}
        require(run['core'] == core and run['native']['polynormer_model_sha256'] == pins['polynormer']['sha256']
                and run['data']['train_npz_sha256'] == pins['train']['sha256']
                and run['data']['valid_npz_sha256'] == pins['development']['sha256'], 'Original canonical core/native/data')
        require(type(saved.get('global')) is bool and type(saved.get('epoch')) is int
                and 1 <= saved['epoch'] <= 1100
                and saved.get('checkpoint_kind') == 'strict-first-maximum complete VALID joint snapshot'
                and model.members == 4 and not model.independent, 'Original canonical selected joint state and stage')
        streams = saved['streams']
        require(len(streams) == 4 and all(set(stream) == {'cpu', 'cuda'} and all(
                value.device.type == 'cpu' and value.dtype == torch.uint8 for value in stream.values())
                for stream in streams), 'Original four saved CPU-byte member RNG streams')
        require(math.isfinite(saved['selected_VALID']) and len(saved['member_VALID']) == 4
                and all(math.isfinite(value) for value in saved['member_VALID']), 'Finite original stored scalar metadata')
        model.load_state_dict(saved['model'], strict=True)
        model.set_global(saved['global'])
        model.eval()
        return streams, dict(epoch=saved['epoch'], global_mode=saved['global'],
            stored_accuracy=saved['selected_VALID'], stored_member_accuracy=saved['member_VALID'],
            method_identity=record['method_identity'], reselection=False, bitwise_parity_claimed=False,
            registered_author_Wiki12_cell=False)
    return restore
