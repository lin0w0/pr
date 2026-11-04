import json
import tempfile
import unittest
from pathlib import Path

from datasets.dataset_split import make_split, save_split, scan_dataset
from datasets.tongji_dataset import read_split_csv
from utils.config import load_dataset_config
from utils.paths import resolve_path


class TestDatasetSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_dataset_config()
        cls.samples = scan_dataset(cls.cfg)
        cls.split = make_split(cls.samples, cls.cfg)

    def test_scan_counts(self):
        self.assertEqual(len(self.samples), 12000)
        self.assertEqual(len({s.label for s in self.samples}), 600)
        for session in ("session1", "session2"):
            sess = [s for s in self.samples if s.session == session]
            self.assertEqual(len(sess), 6000)
            self.assertEqual(len({s.label for s in sess}), 600)

    def test_label_mapping(self):
        by_name = {s.filename: s for s in self.samples}
        self.assertEqual(by_name["00001.bmp"].label, 0)
        self.assertEqual(by_name["00010.bmp"].label, 0)
        self.assertEqual(by_name["00011.bmp"].label, 1)
        self.assertEqual(by_name["00001.bmp"].index_in_id, 0)
        self.assertEqual(by_name["00010.bmp"].index_in_id, 9)
        self.assertEqual(by_name["06000.bmp"].label, 599)
        self.assertEqual(by_name["06000.bmp"].index_in_id, 9)

    def test_protocol_a_counts(self):
        self.assertEqual(len(self.split["gallery"]), 6000)
        self.assertEqual(len(self.split["validation"]), 3000)
        self.assertEqual(len(self.split["test"]), 3000)
        for name in ("gallery", "validation", "test"):
            counts = {}
            for s in self.split[name]:
                counts[s.label] = counts.get(s.label, 0) + 1
            self.assertEqual(len(counts), 600)
            expected = 10 if name == "gallery" else 5
            self.assertEqual(set(counts.values()), {expected})

    def test_no_image_leakage(self):
        sets = {k: {s.path for s in v} for k, v in self.split.items()}
        self.assertFalse(sets["gallery"] & sets["validation"])
        self.assertFalse(sets["gallery"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_all_paths_exist(self):
        for samples in self.split.values():
            for s in samples[:50]:
                self.assertTrue(resolve_path(s.path).is_file(), s.path)

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, meta = save_split(self.split, self.cfg, out_dir=tmp)
            self.assertTrue((Path(out) / "split_meta.json").is_file())
            self.assertEqual(set(meta["files"]), {"gallery", "validation", "test"})
            for name in ("gallery", "validation", "test"):
                loaded = read_split_csv(Path(out) / f"{name}.csv")
                self.assertEqual(len(loaded), len(self.split[name]))
                self.assertEqual(loaded[0].path, self.split[name][0].path)
                self.assertEqual(loaded[-1].label, self.split[name][-1].label)
            with open(Path(out) / "split_meta.json", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["protocol"], "A")


if __name__ == "__main__":
    unittest.main()
