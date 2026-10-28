import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from keyloom_mania.assets import install_bundle
from keyloom_mania.catalog import asset_path, catalog
from keyloom_mania.generation_studio import StudioApp
from tests.helpers import portable_generator


class ReleaseTests(unittest.TestCase):
    def test_catalog_has_only_four_public_families(self):
        self.assertEqual([m['id'] for m in catalog()['generators']],
                         ['music-base', 'count-lookahead', 'count-teacher', 'count-rollout'])
        self.assertTrue(all(m['revision'] == 1 for m in catalog()['generators']))

    def test_asset_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../escape', '/absolute', 'nested/../../escape', 'drive:asset', 'nested\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    asset_path(name, root=directory)

    def test_bundle_verified_before_any_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'installed'
            root.mkdir()
            installed = root / 'model.json'
            installed.write_bytes(b'old')
            content = b'new'
            spec = {'files': {'model.json': {'bytes': 3, 'sha256': hashlib.sha256(content).hexdigest()}}}
            bundle = Path(directory) / 'bundle.zip'
            with patch('keyloom_mania.assets.catalog', return_value=spec):
                for files in ({'../model.json': content}, {'model.json': b'bad'}, {'model.json': content, 'extra': b'x'}):
                    with zipfile.ZipFile(bundle, 'w') as archive:
                        for name, data in files.items():
                            archive.writestr(name, data)
                    with self.assertRaises(ValueError):
                        install_bundle(bundle, root)
                    self.assertEqual(installed.read_bytes(), b'old')
                with zipfile.ZipFile(bundle, 'w') as archive:
                    archive.writestr('model.json', content)
                self.assertEqual(install_bundle(bundle, root), 1)
                self.assertEqual(installed.read_bytes(), content)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_reserves_generation_before_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            source.write_bytes(b'fixture')
            app = StudioApp(root, 8788)
            for identifier in ('a' * 32, 'b' * 32):
                app.jobs[identifier] = {'directory': str(root), 'source_path': str(source), 'filename': source.name, 'status': 'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread'):
                app.generate('a' * 32, {})
                self.assertTrue(app.generation_lock.locked())
                with self.assertRaises(RuntimeError):
                    app.generate('b' * 32, {})
                self.assertEqual(app.jobs['b' * 32]['status'], 'ready')
            with patch('keyloom_mania.generation_studio._generate_song'):
                app._run_generation('a' * 32, source, {})
            self.assertFalse(app.generation_lock.locked())
            self.assertEqual(app.jobs['a' * 32]['status'], 'complete')

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_thread_start_failure_releases_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            app = StudioApp(root, 8788)
            identifier = 'c' * 32
            app.jobs[identifier] = {'directory': str(root), 'source_path': str(source), 'filename': source.name, 'status': 'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                worker.return_value.start.side_effect = RuntimeError('cannot start')
                with self.assertRaises(RuntimeError):
                    app.generate(identifier, {})
            self.assertFalse(app.generation_lock.locked())
            self.assertEqual(app.jobs[identifier]['status'], 'failed')

    def test_web_assets_and_license_are_present(self):
        from keyloom_mania.studio.config import WEB_ROOT
        for name in ('index.html', 'app.js', 'style.css'):
            self.assertGreater((WEB_ROOT / name).stat().st_size, 0)

    def test_public_density_budget_rejects_unsupported_combinations(self):
        from keyloom_mania.generator_density import validate_budget
        for generator, style in [('count-rollout', 'Jumpstream'), ('music-base', 'Minijacks')]:
            with self.assertRaises(ValueError):
                validate_budget('count-reference', for_studio=True, generator_id=generator, main_style=style)
        self.assertEqual(validate_budget('count-reference', for_studio=True,
                         generator_id='music-base', main_style='Jumpstream'), 'count-reference')

    def test_two_live_workbenches_cannot_bind_the_same_address(self):
        from keyloom_mania.generation_studio import StudioHTTPServer, StudioHandler
        server = StudioHTTPServer(('127.0.0.1', 0), StudioHandler)
        try:
            with self.assertRaises(OSError):
                duplicate = StudioHTTPServer(server.server_address, StudioHandler)
                duplicate.server_close()
        finally:
            server.server_close()


if __name__ == '__main__':
    unittest.main()
