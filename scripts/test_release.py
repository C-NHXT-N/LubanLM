"""Release guards: missing inputs, package boundaries and corrupted archives."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from scripts.check_release import inspect


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'.cache/release-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'LubanLM'
        self.root.mkdir()

    def put(self, name, text):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf8')

    def reasons(self):
        return {e['reason'] for e in inspect(self.root)['errors']}

    def test_encoded_space_and_four_backtick_fence(self):
        self.put('README.md', '[appendix](<附录/a%20b.md>)\n\n````text\n[example](missing.md)\n````\n')
        self.put('附录/a b.md', '# Appendix\n')
        self.assertTrue(inspect(self.root)['passed'])

    def test_missing_link(self):
        self.put('README.md', '[missing](missing.md)')
        self.assertIn('link_not_in_package', self.reasons())

    def test_license_texts_are_included_and_scanned(self):
        self.put('LICENSE', 'See LICENSE-CODE and LICENSE-DOCS.')
        self.put('LICENSE-CODE', 'MIT License')
        self.put('LICENSE-DOCS', 'Attribution-ShareAlike 4.0 International')
        self.assertTrue(inspect(self.root)['passed'])
        self.assertEqual(inspect(self.root)['file_count'], 3)
        self.put('LICENSE-DOCS', '-----BEGIN '+'PRIVATE KEY-----')
        self.assertIn('credential_pattern', self.reasons())

    def test_workspace_link(self):
        self.put('README.md', '[author report](../reports/a.json)')
        self.assertIn('link_leaves_package', self.reasons())

    def test_excluded_nested_generated_data(self):
        self.put('docs/run-results/smoke/data/raw/corpus.json', '{}')
        self.put('README.md', '# Tutorial')
        self.assertEqual(inspect(self.root)['file_count'], 1)

    def test_excluded_file_cannot_satisfy_link(self):
        self.put('reports/result.json', '{}')
        self.put('README.md', '[result](reports/result.json)')
        self.assertIn('link_not_in_package', self.reasons())

    def test_syntax_and_credential_guard(self):
        self.put('broken.pyw', 'def missing(:\n')
        self.put('note.txt', '-----BEGIN '+'PRIVATE KEY-----')
        self.assertIn('python_syntax', self.reasons())
        self.assertIn('credential_pattern', self.reasons())

    def test_zip_bytes_must_match_source(self):
        self.put('README.md', '# Tutorial')
        archive = self.base/'changed.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('LubanLM/README.md', '# Different bytes')
        report = self.base/'audit.json'
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('check_release.py')),
            '--root', str(self.root), '--output', str(report), '--zip', str(archive)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        saved = json.loads(report.read_text(encoding='utf8'))
        self.assertIn('zip_sha256_mismatch', {e['reason'] for e in saved['errors']})


if __name__ == '__main__':
    unittest.main()
