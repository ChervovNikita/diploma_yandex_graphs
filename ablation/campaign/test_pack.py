"""Small local archive fixtures, with no conda packing or network access."""
import json
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from .offline import REPOSITORY, record
from .pack import collect_files, environment_archive, repository_archive, verify_repository_archive


class PackagingTests(unittest.TestCase):
    def setUp(self):
        parent = REPOSITORY / '.campaign_test_tmp/pack_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'example.py'
        self.source.write_text('value = 3\n')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(schema='gnnm-campaign-offline-v1',
            files={str(self.source.relative_to(REPOSITORY)): record(self.source)})))

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_archive_checks_every_file_and_excludes_binary_supplements(self):
        reviews = self.root / 'reviews'
        reviews.mkdir()
        (reviews / 'review.md').write_text('Fixture review.\n')
        (reviews / 'unrequested_checkpoint.pt').write_bytes(b'checkpoint fixture')
        files = collect_files(self.manifest, [reviews])
        self.assertEqual(len(files), 3)
        self.assertFalse(any(name.endswith('.pt') for name in files))
        destination = self.root / 'repository.tar.gz'
        receipt = repository_archive(destination, files, 'repository')
        self.assertEqual(receipt['members'], 3)
        self.assertEqual(verify_repository_archive(destination, files, 'repository'), 3)
        with self.assertRaises(FileExistsError):
            repository_archive(destination, files, 'repository')
        changed = dict(files)
        changed[str(self.source.relative_to(REPOSITORY))] = {'bytes': 10, 'sha256': 'incorrect'}
        with self.assertRaises(ValueError):
            verify_repository_archive(destination, changed, 'repository')

    def test_environment_archive_fixture_checks_relocation_helper_and_integrity(self):
        def fake_pack(**kwargs):
            self.assertEqual(kwargs['n_threads'], 2)
            with tarfile.open(kwargs['output'], 'w:gz') as archive:
                for name in ('conda-meta/example.json', 'bin/conda-unpack'):
                    data = b'fixture content'
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        with patch.dict('sys.modules', {'conda_pack': SimpleNamespace(pack=fake_pack)}):
            receipt = environment_archive(self.root, self.root / 'environment.tar.gz', 2)
        self.assertEqual(receipt['members'], 2)
        self.assertTrue((self.root / 'environment.tar.gz').is_file())
        self.assertFalse((self.root / '.environment.tar.gz.building').exists())

    def test_changed_manifest_input_and_explicit_binary_supplement_are_rejected(self):
        binary = self.root / 'unrequested.pt'
        binary.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'text evidence'):
            collect_files(self.manifest, [binary])
        self.source.write_text('changed = True\n')
        with self.assertRaisesRegex(ValueError, 'changed or is missing'):
            collect_files(self.manifest, [])


if __name__ == '__main__':
    unittest.main()
