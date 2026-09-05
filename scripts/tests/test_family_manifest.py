"""Exercise candidate isolation and the real family validation failure paths."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_family_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_family_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FAMILY_SOURCE = Path(os.environ["FAMILY_TEST_ROOT"])


class FamilyManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.family = self.root / "family"
        self.family.mkdir()
        (self.family / "scripts").mkdir()
        for name in ("check_compatibility.py", "check_artifact_provenance.py"):
            shutil.copyfile(FAMILY_SOURCE / "scripts" / name, self.family / "scripts" / name)
        self.git("init", "--quiet")
        self.base = {"compatibility": {"schema": 1}}
        gitmodules, manifest = [], ["[compatibility]\nschema = 1\n"]
        for index, name in enumerate(MODULE.PACKAGES, start=1):
            revision = str(index) * 40
            self.base[name] = {"repo": f"gomeshun/{name}", "path": name, "ref": revision}
            manifest.append(f'[{name}]\nrepo = "gomeshun/{name}"\npath = "{name}"\nref = "{revision}"\n')
            gitmodules.append(f'[submodule "{name}"]\n\tpath = {name}\n\turl = git@github.com:gomeshun/{name}.git\n')
            self.git("update-index", "--add", "--cacheinfo", f"160000,{revision},{name}")
        (self.family / "compatibility.toml").write_text("\n".join(manifest))
        (self.family / ".gitmodules").write_text("\n".join(gitmodules))
        self.git("add", "compatibility.toml", ".gitmodules", "scripts")
        self.git("-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "--quiet", "-m", "test base")
        self.output = self.root / "effective.toml"
        self.candidate = "a" * 40

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.family), *args], text=True).strip()

    def prepare(self, **kwargs):
        return MODULE.prepare(self.family, self.output, "sashimi-f", self.candidate, **kwargs)

    def test_candidate_only_replaces_one_revision_and_records_base(self):
        original = (self.family / "compatibility.toml").read_bytes()
        base_ref, effective = self.prepare()
        self.assertEqual(base_ref, self.git("rev-parse", "HEAD"))
        self.assertEqual(effective["sashimi-f"]["ref"], self.candidate)
        for name in MODULE.PACKAGES[:-1]:
            self.assertEqual(effective[name], self.base[name])
        self.assertEqual(tomllib.loads(self.output.read_text()), effective)
        self.assertEqual((self.family / "compatibility.toml").read_bytes(), original)
        record = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(record["family_base_ref"], base_ref)
        self.assertEqual(record["effective_revisions"]["sashimi-f"], self.candidate)

    def test_promoted_mode_keeps_every_canonical_revision(self):
        _, effective = self.prepare(mode="promoted")
        self.assertEqual(effective, self.base)

    def test_c_candidate_keeps_f_and_other_sibling_revisions(self):
        _, effective = MODULE.prepare(self.family, self.output, "sashimi-c", self.candidate)
        self.assertEqual(effective["sashimi-c"]["ref"], self.candidate)
        for name in MODULE.PACKAGES:
            if name != "sashimi-c":
                self.assertEqual(effective[name], self.base[name])

    def test_refuses_canonical_output_path(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            MODULE.prepare(self.family, self.family / "compatibility.toml", "sashimi-f", self.candidate)

    def test_rejects_non_immutable_candidate_ref(self):
        for value in ("main", "abc123", "A" * 40, "a" * 39 + "\n"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "full lowercase"):
                MODULE.prepare(self.family, self.output, "sashimi-f", value)

    def test_non_candidate_gitlink_mismatch_fails_before_output(self):
        self.git("update-index", "--cacheinfo", "160000," + "b" * 40 + ",sashimi-si")
        self.git("-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "--quiet", "-m", "mismatched SI")
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_candidate_base_mismatch_is_not_hidden_by_override(self):
        path = self.family / "compatibility.toml"
        path.write_text(path.read_text().replace("5" * 40, "c" * 40))
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_candidate_wheel_provenance_is_rejected(self):
        self.prepare()
        artifact = self.root / "sashimi_f-0.1.0a1-py3-none-any.whl"
        checker = self.family / "scripts/check_artifact_provenance.py"
        import sys

        command = [sys.executable, str(checker), "--directory", str(self.root),
                   "--manifest", str(self.output), "--packages", "sashimi-f"]
        for revision, expected_code in (("5" * 40, 1), (self.candidate, 0)):
            with zipfile.ZipFile(artifact, "w") as archive:
                archive.writestr("_sashimi_f_build_provenance.py", f'SOURCE_REVISION = "{revision}"\n')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
