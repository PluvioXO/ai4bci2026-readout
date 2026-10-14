"""Small checks for the new runner's numerical and data-separation boundaries."""
import ast
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import replicate_readout as r


class ReplicationChecks(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        r.configure_engine()

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_filter_order_gradients_and_inference(self):
        r.self_check()

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = replace(r.b.smoke_config(), subjects=(1,), epochs=3)
            data = r.b.prepare_subject(cfg, 1, root / "data", root, target=False)
            setting = r.b.candidate_settings(cfg, "ordinary")[0]
            full = r.b.fit_model(cfg, "ordinary", 1, 0, setting, data, root / "full", torch.device("cpu"))
            partial = r.b.fit_model(cfg, "ordinary", 1, 0, setting, data, root / "resume",
                                    torch.device("cpu"), stop_after=1)
            self.assertIsNone(partial)
            resumed = r.b.fit_model(cfg, "ordinary", 1, 0, setting, data, root / "resume", torch.device("cpu"))
            for key, value in full[0].state_dict().items():
                self.assertTrue(torch.equal(value, resumed[0].state_dict()[key]))
            self.assertEqual(full[2], resumed[2])

    def test_target_is_loaded_only_after_source_choices_are_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = replace(r.b.smoke_config(), subjects=(1,))
            prepare = r.b.prepare_subject
            target_calls = []
            def checked(config, subject, cache, root, target=False):
                if target:
                    locked = json.loads((Path(root) / "source_locked.json").read_text())
                    self.assertEqual(len(locked), 5)  # pilot + main weights, optimizer choice, head and head choice
                    self.assertTrue(all(r.b.sha_file(Path(root) / p) == sha for p, sha in locked.items()))
                    target_calls.append(subject)
                return prepare(config, subject, cache, root, target=target)
            with patch.object(r.b, "prepare_subject", checked):
                result = r.run(cfg, Path(tmp) / "data", Path(tmp) / "runs")
            self.assertEqual(target_calls, [1])
            self.assertTrue(json.loads((result / "complete.json").read_text())["synthetic"])

    def test_replication_notebook_embeds_exact_checked_sources(self):
        nb = json.loads((ROOT / "ai4bci_readout_replication.ipynb").read_text())
        sources = None
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == "SOURCES" for n in node.targets):
                    sources = ast.literal_eval(node.value)
        self.assertIsNotNone(sources)
        for name, source in sources.items():
            self.assertEqual(source, (ROOT / "code" / name).read_text())
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), nb["metadata"]["ai4bci_source_sha256"][name])


if __name__ == "__main__":
    unittest.main()
