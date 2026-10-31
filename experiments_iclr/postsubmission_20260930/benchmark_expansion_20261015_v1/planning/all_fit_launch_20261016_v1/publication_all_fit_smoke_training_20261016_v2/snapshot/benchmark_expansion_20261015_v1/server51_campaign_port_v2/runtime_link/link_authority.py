"""Exact source-bound link extension declarations; no model or tensor imports.

A registry entry describes the authentic frozen producer API. It does not qualify
training, resources, devices or campaign authority. Unknown declarations fail.
"""
from __future__ import annotations
import copy
from pathlib import Path

REGISTRY = {'interaction_link_ncn_v2': {'callback_name': 'callback.py', 'callback_sha256': '1ba45e9757024e4a61f2edb81d93288643271a27b2255932e5201b169c25f225', 'declaration_path': ['interaction_metadata'], 'schema': 'explicit-NCN-interaction-v1', 'extension_name': 'session.py', 'mean': 'provider', 'manifest_sha256': '8f6cc6ca60214b47653ba4c305f2b98dfae503c08ae52ee32b30de67ed132450'}, 'interaction_link_ncnc_v1': {'callback_name': 'callback.py', 'callback_sha256': 'f80ff4b8a8ce5b285cdb5cdcd81825b3cee03f57a67ccba7dcc32f54dd8f4c2b', 'declaration_path': ['interaction_metadata'], 'schema': 'explicit-NCNC-interaction-v1', 'extension_name': 'session.py', 'mean': 'provider', 'manifest_sha256': 'e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1'}, 'interaction_BUDDY_production_v2': {'callback_name': 'callback.py', 'callback_sha256': '0381834e7d0c66677b6a1acbf3ecc6ebd41fe2e406bd598d9d3cd1bd40ff8668', 'declaration_path': ['interaction_metadata'], 'schema': 'explicit-BUDDY-interaction-production-v2', 'extension_name': 'session.py', 'mean': 'provider', 'manifest_sha256': 'eae2516bf1ebd191237ba5e6e72b0d84d619f01f390068abaadb19e46cad8818'}, 'link_expansion_ports_v1': {'callback_name': 'callback.py', 'callback_sha256': '018c2f231983c4592b42b40d436ac3e0627f2ad29f4225bf9b711b61b574c0f1', 'declaration_path': ['method_metadata'], 'schema': 'native-NCN-expanded-risk-v1', 'extension_name': 'session.py', 'mean': 'C6', 'manifest_sha256': 'c80f6e736e6cb96683f5f429ab9a3c3f78c19c8ed241e074a741f54e9d18c8db'}, 'link_expansion_ports_NCNC_v1': {'callback_name': 'callback.py', 'callback_sha256': '308b9064c4382b54bd571fdd5f99f15fb4bb66a9743ef58f424b83f7da70a220', 'declaration_path': ['method_metadata'], 'schema': 'native-NCNC-expanded-risk-v1', 'extension_name': 'session.py', 'mean': 'C6', 'manifest_sha256': '7621ae9e64c110480a93a441b2853ea60c2e3febb9bd9063197e42d5feaec88b'}, 'link_expansion_ports_BUDDY_production_v1': {'callback_name': 'callback.py', 'callback_sha256': 'de81b890314f74bf968e75977cbaa60e08c0b1c4ab0ad00f6394980eefaf651e', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-expanded-risk-production-v1', 'extension_name': 'session.py', 'mean': 'C6', 'manifest_sha256': '7cf3d5d7bf67a89400276341b353c31f25e6e59861ba2f2058c3f9fff853c534'}, 'context_affinity_link_port_v2': {'callback_name': 'callback.py', 'callback_sha256': 'c902b13465b8acddba8491ec07aade12610d09efb5bd20ef8bb42f56bb5d0bba', 'declaration_path': ['method_metadata'], 'schema': 'context-affinity-native-NCN-v1', 'extension_name': 'session.py', 'mean': 'none', 'manifest_sha256': '4a43ff70bbe99e5d1db0c26972866abdc260de43c3791dabf6215405a5e79b2e'}, 'context_affinity_ncnc_port_v1': {'callback_name': 'callback.py', 'callback_sha256': '15ac40e9838fab7a929329c138d364b6a33ea8caeadedaf5ca00ec1dee0801ab', 'declaration_path': ['method_metadata'], 'schema': 'context-affinity-native-NCNC-v1', 'extension_name': 'session.py', 'mean': 'none', 'manifest_sha256': '1748b647dee6de992261dfc96b8d1b0b210307d786ce61b7da4f7d05a1b0b526'}, 'context_affinity_BUDDY_port_v1': {'callback_name': 'callback.py', 'callback_sha256': '712ac7dfcbb6affb7c28ef7b27abc1af3ecf30c12b4e4e961dc0bf8306443685', 'declaration_path': ['method_metadata'], 'schema': 'context-affinity-native-BUDDY-v1', 'extension_name': 'session.py', 'mean': 'none', 'manifest_sha256': '4b15c6fae17d668bf4c567e223b42ca32f24582cc081dfd2c0c90f0a9835d64d'}, 'portable_private_input_v2': {'callback_name': 'link_callback.py', 'callback_sha256': '1a815208f2d2e10945af1eced5815a8c4a2eb613fdb0bf89c52dc4bb54bf772b', 'declaration_path': ['native', 'portable_private_input_metadata'], 'schema': 'portable-private-input-session-v1', 'extension_name': 'extension.py', 'mean': 'none', 'manifest_sha256': '21ad8f2c644a1833f0015dfc4c0c0193bcfc9598b6f84d98b9d686330bed8115'}, 'buddy_private_bn_production_v1': {'callback_name': 'callback.py', 'callback_sha256': '85896f0f838bf966f4093899f35a51582de2ece612bb17152eb8c378a051fd6e', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-private-BN-production-source-v1', 'extension_name': 'session.py', 'mean': 'same_call', 'manifest_sha256': 'a9d267c09b07b512da0be6e00a3b690552b58a6ad19e0c14ba29291c35ab6581'}, 'interaction_BUDDY_production_v3': {'callback_name': 'callback.py', 'callback_sha256': '8a656386a04c0b90a49769ad42083e6722bfd51458ae50e7ea0e3c0e55a9db31', 'declaration_path': ['interaction_metadata'], 'schema': 'explicit-BUDDY-interaction-production-v2', 'extension_name': 'session.py', 'mean': 'provider', 'manifest_sha256': '78b0f9b00aa34727356c2a99787dfc3e6f1b4a43f380c3d7194b9179b4b4550b'}, 'link_expansion_ports_BUDDY_production_v2': {'callback_name': 'callback.py', 'callback_sha256': 'dcf6e28aec3d4620b31d170cf11924446b3ee707ed546e8f4eb9805ccf407c00', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-expanded-risk-production-v1', 'extension_name': 'session.py', 'mean': 'C6', 'manifest_sha256': '7ff6c81f00daed03ac4be99d4d011dc47305aa78bcc94d97c4220715ff394acd'}, 'buddy_private_bn_production_v2': {'callback_name': 'callback.py', 'callback_sha256': '9d3c427cfc96f59dd71381ae53186eeaa2c8cef087ecb004bbd89852c6840a73', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-private-BN-production-source-v1', 'extension_name': 'session.py', 'mean': 'same_call', 'manifest_sha256': '3e507e04fb307560d082b087bf7aabc68203614b0245c5e07a538353535a34f0'}}
C6 = ('aggregation_mean_globalJ_v1', 'aggregation_global_globalJ_v1', 'aggregation_conditional_globalJ_v1')
MEAN_KEYS = ('same_bank_mean_outputs', 'same_bank_mean_loss', 'same_bank_mean_metrics')


def authenticate_extension(session, job, extension, root, receipt, inside):
    callback = inside(job['callback_module'], root)
    entry = REGISTRY.get(callback.parent.name)
    if entry is None or callback.name != entry['callback_name'] or receipt(callback)['sha256'] != entry['callback_sha256'] \
            or job.get('callback_sha256') != entry['callback_sha256']:
        raise ValueError('Unreviewed link extension callback/source')
    expected_extension = inside(callback.with_name(entry['extension_name']), root)
    if extension != receipt(expected_extension):
        raise ValueError('Link extension path or exact source receipt differs')
    manifest = inside(callback.with_name('MANIFEST.json'), root)
    manifest_pin = receipt(manifest)
    if job.get('method_settings', {}).get('manifest_sha256') != entry['manifest_sha256'] \
            or manifest_pin['sha256'] != entry['manifest_sha256']:
        raise ValueError('Explicit outer link method manifest differs')
    declaration = session
    for attribute in entry['declaration_path']:
        declaration = getattr(declaration, attribute, None)
        if declaration is None:
            raise ValueError('Authentic link declaration attribute is missing')
    if not isinstance(declaration, dict) or declaration.get('schema') != entry['schema'] \
            or declaration.get('method_id') != job['procedure'] or declaration.get('TEST_accessed') is not False:
        raise ValueError('Link declaration schema/method/closed TEST authority differs')
    pins = declaration.get('source_receipts')
    if not isinstance(pins, list) or not pins:
        raise ValueError('Explicit authentic link method source receipts required')
    unique = {}
    for pin in pins:
        if set(pin) != {'path','bytes','sha256'} or receipt(inside(pin['path'], root)) != pin \
                or (pin['path'] in unique and unique[pin['path']] != pin):
            raise ValueError('Link method source receipt differs or has unrecognized annotation')
        unique[pin['path']] = pin
    if unique.get(str(manifest)) != manifest_pin or unique.get(str(expected_extension)) != extension:
        raise ValueError('Outer manifest/extension absent from authentic source closure')
    mean = entry['mean']
    if mean == 'C6':
        mean = 'same_call' if job['procedure'] in C6 else 'none'
        if (declaration.get('aggregation') is not None) != (mean == 'same_call'):
            raise ValueError('C6 method/readout declaration differs')
    return {'schema':'authenticated-link-extension-authority-v1', 'declaration':copy.deepcopy(declaration),
        'mean_protocol':mean, 'callback':receipt(callback), 'extension':copy.deepcopy(extension),
        'manifest':manifest_pin, 'source_receipts':copy.deepcopy(pins)}

# Explicit typed-provider successor identities; all historical entries remain.
REGISTRY.update({'link_expansion_ports_BUDDY_production_v3': {'callback_name': 'callback.py', 'callback_sha256': 'ed9604bb0efee14a3f057e9c68bc255a66b0fe28bc6087c5064b6597cd0d9e11', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-expanded-risk-production-v1', 'extension_name': 'session.py', 'mean': 'C6', 'manifest_sha256': '13e2db4d9cc58755caece9d76504d3e72643554bfa9bc2290d101bc1fa66ac2b'}, 'interaction_BUDDY_production_v4': {'callback_name': 'callback.py', 'callback_sha256': 'fc3db9a53c707f48e64885f33e1e152fea8e06121f8280b8733a2bf78cf1b843', 'declaration_path': ['interaction_metadata'], 'schema': 'explicit-BUDDY-interaction-production-v2', 'extension_name': 'session.py', 'mean': 'provider', 'manifest_sha256': '540c2bf40f6d1a5ae14ea77a77ce5d804aaba9203262314a785945423031a1de'}, 'buddy_private_bn_production_v4': {'callback_name': 'callback.py', 'callback_sha256': '489d66ed509aac57ad1cb4f841e442cbbb29dd36893027b1145cb0406b694108', 'declaration_path': ['method_metadata'], 'schema': 'native-BUDDY-private-BN-production-source-v1', 'extension_name': 'session.py', 'mean': 'same_call', 'manifest_sha256': '483465cd0837f20ea3776bc2efb53e9af87f1ec90a810b5cccc50694b2392771'}})
