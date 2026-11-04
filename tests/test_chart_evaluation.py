import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from keyloom_mania.evaluation import build_reference, evaluate_chart, load_chart, load_reference, render_comparison, render_markdown
from keyloom_mania.evaluation.charts import chart_rows, parse_chart_text
from keyloom_mania.evaluation.features import GAP_VALUES, HISTOGRAMS, combine, extract_blocks
from keyloom_mania.evaluation.audio import SAMPLE_RATE, alignment_metrics
from keyloom_mania.evaluation.reference import manifest_entries
from keyloom_mania.evaluation.scoring import human_likeness, js_divergence, typicality
from keyloom_mania.timing import BeatClock, TimingSection


def osu_text(rows, bpm=150., creator="Human mapper", version="Test", holds=()):
    beat = 60_000 / bpm
    objects = []
    for index, (position, mask) in enumerate(rows):
        time_ms = round(1000 + position * beat)
        for column in range(4):
            if mask >> column & 1:
                x = 64 + 128 * column
                if (index, column) in holds:
                    objects.append(f"{x},192,{time_ms},128,0,{time_ms + 200}:0:0:0:0:")
                else:
                    objects.append(f"{x},192,{time_ms},1,0,0:0:0:0:")
    return ("osu file format v14\n\n[General]\nAudioFilename: audio.mp3\nMode: 3\n\n[Metadata]\nTitle:Fixture\nArtist:Test\n"
            f"Creator:{creator}\nVersion:{version}\n\n[Difficulty]\nHPDrainRate:8\nCircleSize:4\nOverallDifficulty:8\n\n"
            f"[TimingPoints]\n1000,{beat},4,1,0,100,1,0\n\n[HitObjects]\n" + "\n".join(objects) + "\n")


def human_like_rows(seed, bars=48):
    rng = np.random.default_rng(seed)
    rows, previous = [], 0
    for step in range(bars * 16):
        if rng.random() < .15:
            continue
        size = rng.choice((1, 2, 3), p=(.55, .38, .07))
        options = [mask for mask in range(1, 16) if mask.bit_count() == size and not mask & previous]
        mask = int(rng.choice(options)) if options else int(rng.integers(1, 16))
        rows.append((step / 4, mask))
        previous = mask
    return rows


class ChartParsingTests(unittest.TestCase):
    def test_rounded_chords_merge_and_duplicates_are_reported(self):
        text = osu_text([(0, 1), (1, 2)]).replace("1,192,1000,1", "1,192,1000,1")
        text = text.rstrip("\n") + "\n320,192,1001,1,0,0:0:0:0:\n64,192,1000,1,0,0:0:0:0:\n"
        _, sections, notes, issues = parse_chart_text(text)
        times, masks, _ = chart_rows(notes)
        self.assertEqual(len(sections), 1)
        self.assertEqual(masks[0], 0b101)
        self.assertEqual(len(times), 2)
        self.assertIn("duplicate_note", issues)


class FeatureTests(unittest.TestCase):
    def _blocks(self, rows, bpm=120.):
        _, sections, notes, _ = parse_chart_text(osu_text(rows, bpm))
        times, masks, holds = chart_rows(notes)
        return combine(extract_blocks(BeatClock(sections), times, masks, holds))

    def test_hand_observations_forget_everything_older_than_four_beats(self):
        # A hit eight beats ago is indistinguishable from no previous hit at all.
        segment = self._blocks([(0, 1), (8, 1)])
        self.assertEqual(segment["hist"]["same_column_gap"][len(GAP_VALUES)], 2)
        self.assertEqual(segment["hist"]["hand_notes_4b"][1], 2)
        self.assertEqual(segment["hist"]["hand_run"][1], 2)
        near = self._blocks([(0, 1), (2, 1)])
        self.assertEqual(near["hist"]["same_column_gap"][GAP_VALUES.index(2.)], 1)
        self.assertEqual(near["hist"]["hand_notes_4b"][2], 1)
        self.assertEqual(near["hist"]["hand_notes_2b"][1], 2)

    def test_alternating_stream_two_beat_load(self):
        order = (1, 4, 2, 8)
        segment = self._blocks([(step / 4, order[step % 4]) for step in range(64)])
        rows_2b = np.asarray(segment["hist"]["hand_rows_2b"])
        # Half-open two-beat windows hold eight sixteenths, four per alternating hand.
        self.assertEqual(int(rows_2b.argmax()), 4)
        self.assertEqual(segment["hist"]["chord_size"][1], 64)
        self.assertEqual(sum(segment["hist"]["hand_bias_4b"]), 64)

    def test_histogram_shapes_are_stable(self):
        segment = self._blocks(human_like_rows(1, bars=20))
        for name, bins in HISTOGRAMS.items():
            self.assertEqual(len(segment["hist"][name]), bins)
        self.assertGreater(segment["scalars"]["motif_observations"], 0)


class CalibrationTests(unittest.TestCase):
    def test_typicality_is_two_sided_and_one_sided_by_kind(self):
        samples = list(np.linspace(0, 1, 101))
        self.assertEqual(typicality(.5, samples, "two_sided")["tail"], 1.)
        self.assertEqual(typicality(-1, samples, "two_sided")["status"], "atypical")
        self.assertEqual(typicality(2, samples, "two_sided")["status"], "atypical")
        self.assertEqual(typicality(2, samples, "low_bad")["tail"], 1.)
        self.assertEqual(typicality(2, samples, "divergence")["tail"], 0.)
        self.assertIsNone(typicality(.5, samples[:3], "two_sided"))

    def test_human_likeness_is_uniform_for_humans_and_low_for_outliers(self):
        rng = np.random.default_rng(3)
        rows = [{"hand_rows_2b_mean": float(a), "js:transition": float(b)} for a, b in rng.normal(size=(300, 2))]
        names = ["hand_rows_2b_mean", "js:transition"]
        central, _ = human_likeness({"hand_rows_2b_mean": 0., "js:transition": -3.}, rows, names)
        outlier, _ = human_likeness({"hand_rows_2b_mean": 4., "js:transition": 3.}, rows, names)
        self.assertGreater(central, 90)
        self.assertLess(outlier, 2)
        scores = [human_likeness(row, rows[:index] + rows[index + 1:], names)[0] for index, row in enumerate(rows[:60])]
        self.assertTrue(35 < np.mean(scores) < 65)

    def test_jensen_shannon_bounds(self):
        self.assertAlmostEqual(js_divergence([5, 5], [5, 5], prior=0), 0.)
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1], prior=0), 1.)
        self.assertAlmostEqual(js_divergence([3, 1], [1, 3]), js_divergence([1, 3], [3, 1]))


def click_track(bpm=120., beats=128, strong=1., weak=.12, offset_ms=1000):
    """Noise bursts: a strong one on every beat and a weak one on every other sixteenth."""
    rng = np.random.default_rng(5)
    beat = 60_000 / bpm
    waveform = np.zeros(int(SAMPLE_RATE * (offset_ms + beat * (beats + 2)) / 1000), dtype=np.float32)
    for step in range(beats * 4):
        amplitude = strong if step % 4 == 0 else weak
        start = int(SAMPLE_RATE * (offset_ms + step * beat / 4) / 1000)
        waveform[start:start + 120] += amplitude * rng.standard_normal(120).astype(np.float32)
    return waveform


class AccentTests(unittest.TestCase):
    def _metrics(self, rows, waveform):
        _, sections, notes, _ = parse_chart_text(osu_text(rows, bpm=120.))
        times, masks, _ = chart_rows(notes)
        return alignment_metrics(waveform, BeatClock(sections), times, masks, np.ones(len(times), dtype=bool))

    def test_chords_on_strong_beats_raise_accent_lift(self):
        waveform = click_track()
        on_accents = [(step / 4, 0b0101 if step % 4 == 0 else 1 << (step % 3 + 1)) for step in range(120 * 4)]
        off_accents = [(step / 4, 0b0101 if step % 4 == 2 else 1 << (step % 3 + 1)) for step in range(120 * 4)]
        good, bad = self._metrics(on_accents, waveform), self._metrics(off_accents, waveform)
        self.assertGreater(good["accent_lift"], 2)
        self.assertLess(bad["accent_lift"], 1)
        self.assertGreater(good["size_strength_spearman"], bad["size_strength_spearman"])
        self.assertGreater(good["strong_accent_chord_recall"], .9)

    def test_chords_without_onsets_are_unsupported(self):
        waveform = click_track()
        # Chords on 1/8 positions between the sixteenth clicks have no onset within 30 ms.
        rows = [(step / 4, 1 << (step % 4)) for step in range(120 * 4)] + [(step + .125, 0b1001) for step in range(120)]
        result = self._metrics(sorted(rows), waveform)
        self.assertGreater(result["unsupported_chords"], .9)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        corpus = root / "corpus"
        corpus.mkdir()
        for seed in range(16):
            (corpus / f"human-{seed}.osu").write_text(osu_text(human_like_rows(seed)), encoding="utf-8")
        (corpus / "generated.osu").write_text(osu_text(human_like_rows(99), creator="Keyloom generator/music-base"), encoding="utf-8")
        cls.reference_path = root / "reference.json.gz"
        cls.profile = build_reference([corpus], cls.reference_path, workers=1, progress=lambda *_: None)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_reference_excludes_generated_charts_and_keeps_no_chart_text(self):
        self.assertEqual(self.profile["summary"]["accepted"], 16)
        self.assertEqual(self.profile["summary"]["skipped"], {"generated_or_research_chart": 1})
        with gzip.open(self.reference_path, "rt", encoding="utf-8") as source:
            content = source.read()
        self.assertNotIn("HitObjects", content)
        self.assertNotIn(str(self.root), content)

    def test_manifest_reads_only_requested_splits(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps({"split": split, "chart_relative_path": f"corpus/human-{index}.osu",
                                                   "audio_relative_path": f"audio-{index}.wav"})
                                       for index, split in enumerate(("train", "validation", "test", "quarantine"))), encoding="utf-8")
        entries = manifest_entries(manifest)
        self.assertEqual(len(entries), 1)
        self.assertTrue(str(entries[0][0]).endswith("human-0.osu"))
        self.assertEqual(len(manifest_entries(manifest, ("train", "validation"))), 2)

    def test_mismatched_feature_schema_is_rejected(self):
        stale = self.root / "stale.json.gz"
        with gzip.open(stale, "wt", encoding="utf-8") as target:
            json.dump({**self.profile, "feature_schema": "pressure-2-8-32"}, target)
        with self.assertRaises(ValueError):
            load_reference(stale)

    def test_human_charts_score_in_range_and_jack_wall_does_not(self):
        reference = load_reference(self.reference_path)
        human = []
        for seed in range(6):
            # Corpus charts are excluded from their own neighbours, so these are leave-one-out humans.
            with load_chart(self.root / "corpus" / f"human-{seed}.osu") as chart:
                human.append(evaluate_chart(chart, reference, with_audio=False, neighbor_count=15))
        wall = self.root / "wall.osu"
        wall.write_text(osu_text([(step / 4, 1) for step in range(48 * 16)], version="wall"), encoding="utf-8")
        with load_chart(wall) as chart:
            bad = evaluate_chart(chart, reference, with_audio=False, neighbor_count=15)
        scores = [report["overall"]["naturalness"] for report in human]
        self.assertGreater(np.mean(scores), 25)
        within = [report["overall"]["style_naturalness"] for report in human if report["overall"]["style_naturalness"] is not None]
        self.assertTrue(within)
        self.assertTrue(any(name.startswith("style:") for name in human[0]["metrics"]))
        self.assertLess(bad["overall"]["naturalness"], 5)
        self.assertLess(bad["axes"]["hand_pressure"]["score"], np.mean([r["axes"]["hand_pressure"]["score"] for r in human]))
        self.assertIn("js:column_hits_4b", bad["axes"]["hand_pressure"]["concerns"])
        self.assertTrue(bad["validity"]["passed"])
        self.assertIn("手部压力", render_markdown(bad))
        self.assertIn("对比", render_comparison([human[0], bad]))

    def test_package_plan_adherence_and_execution_stay_visible(self):
        reference = load_reference(self.reference_path)
        rows = human_like_rows(41, bars=16)
        beat = 400.
        bars = [{"bar": index + 1, "start_ms": 1000 + index * 4 * beat, "end_ms": 1000 + (index + 1) * 4 * beat,
                 "style": "Minijacks" if index == 3 else "Jumpstream", "quiet": False} for index in range(16)]
        package = self.root / "candidate.osz"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("generated.osu", osu_text(rows))
            archive.writestr("bar-density.json", json.dumps({"bars": bars}))
            archive.writestr("density-execution.json", json.dumps({"summary": {"music_notes": 10, "export_notes": 9, "export_removed_notes": 1}}))
            archive.writestr("export-provenance.json", json.dumps({"settings": {"target_star": 5.0}}))
        with load_chart(package) as chart:
            report = evaluate_chart(chart, reference, with_audio=False, neighbor_count=10)
        self.assertEqual(report["plan_adherence"]["eligible_bars"], 16)
        self.assertIn("rule", report["plan_adherence"]["per_style"]["Jumpstream"])
        self.assertEqual(report["plan_adherence"]["passed_bars"] + len(report["plan_adherence"]["failed_bars"]), 16)
        self.assertIsNotNone(report["validity"]["minijack_audit"])
        self.assertEqual(report["execution"]["downstream_losses"], 1)
        self.assertEqual(report["difficulty"]["target_star"], 5.0)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
