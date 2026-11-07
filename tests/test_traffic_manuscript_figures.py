"""Editorial figures preserve the selected evidence and disclose denominators."""

import hashlib
import importlib.util
import json
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import pandas as pd

if importlib.util.find_spec("scipy") is None:
    raise unittest.SkipTest("Research dependencies required")

HERE = Path(__file__).resolve().parents[1] / "research/2026-10/traffic"
sys.path.insert(0, str(HERE))


class ManuscriptFigures(unittest.TestCase):
    def test_resolved_noise_grid_and_original_anchors(self):
        from analyze_noise_resolution import ANCHORS, SPEED_NOISE, seeds

        folder = HERE / "results/noise_resolution"
        data = pd.read_csv(folder / "diagnostics.csv")
        models = pd.read_csv(folder / "operators.csv")
        summary = pd.read_csv(folder / "summary.csv")
        self.assertEqual(len(models), 2 * 11 * 20)
        self.assertTrue(models.status.eq("eligible").all())
        self.assertEqual(len(data), 2 * 11 * 20 * 16)
        self.assertEqual(len(summary), 2 * 11 * 4)
        self.assertTrue(summary["count"].eq(80).all())
        np.testing.assert_array_equal(sorted(data.speed_noise_sd_mps.unique()), SPEED_NOISE)
        self.assertTrue(data.rejection_fraction.between(0, 1).all())
        np.testing.assert_allclose(
            data.rejection_fraction * 100, np.round(data.rejection_fraction * 100), atol=1e-12
        )
        for row in models.itertuples():
            actual = (row.training_seed, row.calibration_seed, row.evaluation_seed)
            self.assertEqual(
                actual,
                seeds(
                    [1, 1.8].index(row.tau),
                    SPEED_NOISE.index(row.speed_noise_sd_mps),
                    row.training_replicate,
                ),
            )
        used = models[["training_seed", "calibration_seed", "evaluation_seed"]].to_numpy().ravel()
        self.assertEqual(len(np.unique(used)), len(used))
        recomputed = data.groupby(
            ["tau", "dt", "excitation", "speed_noise_sd_mps"]
        ).rejection_fraction.agg(["mean", "min", "max", "count"])
        pd.testing.assert_frame_equal(
            summary.set_index(["tau", "dt", "excitation", "speed_noise_sd_mps"]), recomputed
        )
        original = pd.read_csv(HERE / "results/uncertainty/physics_fit/diagnostics.csv")
        original = original[
            (original.method == "restricted_idm") & (original.calibration == "direct_calibration")
        ].copy()
        original["speed_noise_sd_mps"] = original.noise_index.map(dict(enumerate(ANCHORS)))
        anchor = data.merge(
            original,
            on=["id", "speed_noise_sd_mps", "training_replicate"],
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        self.assertEqual(len(anchor), 1920)
        for column in ["cutoff", "mean_score", "rejection_fraction"]:
            np.testing.assert_allclose(
                anchor[column + "_new"], anchor[column + "_old"], rtol=1e-12, atol=1e-10
            )
        receipt = json.loads((folder / "independent_validation.json").read_text())
        self.assertEqual(receipt["fits_replayed"], 22)
        self.assertEqual(receipt["scalar_comparisons"], 726)
        self.assertLess(receipt["max_absolute_error"], 1e-9)

    def test_consistent_palette_and_parameter_legend(self):
        import matplotlib.pyplot as plt
        import render_manuscript_figures as renderer
        from matplotlib.colors import to_hex

        self.assertEqual(to_hex(renderer.SIGNED_SPEED(0.0)), renderer.BLUE)
        self.assertEqual(to_hex(renderer.SIGNED_SPEED(1.0)), renderer.GOLD)
        self.assertEqual(to_hex(renderer.ABSOLUTE_SPEED(1.0)), renderer.BLUE)
        with patch.object(renderer, "save") as save:
            renderer.detection()
        fig, name = save.call_args.args
        try:
            self.assertEqual(name, "observation_limits")
            ax = fig.axes[0]
            for line, color in zip(
                ax.lines[:4], [renderer.BLUE, renderer.BLUE, renderer.GOLD, renderer.GOLD]
            ):
                self.assertEqual(line.get_color(), color)
                self.assertEqual(len(line.get_xdata()), 11)
            labels = [label.get_text() for label in ax.get_legend().get_texts()]
            self.assertEqual(len(labels), 4)
            for label in labels:
                self.assertIn(r"\tau=", label)
                self.assertIn(r"\epsilon=", label)
                self.assertIn(r"\mathrm{s}", label)
                self.assertIn(r"\mathrm{m\,s^{-1}}", label)
                self.assertNotIn("Headway", label)
            self.assertIn(r"\sigma_v", ax.get_xlabel())
            self.assertIn(r"10^{-3}", ax.get_xticklabels()[0].get_text())
            fig.canvas.draw()
            legend_box = ax.get_legend().get_window_extent()
            axes_box = ax.get_window_extent()
            self.assertGreaterEqual(legend_box.x0, axes_box.x0)
            self.assertLessEqual(legend_box.x1, axes_box.x1)
            self.assertGreaterEqual(legend_box.y0, axes_box.y0)
            self.assertLessEqual(legend_box.y1, axes_box.y1)
            for line in ax.lines[:4]:
                self.assertFalse(
                    line.get_path()
                    .transformed(ax.transData)
                    .intersects_bbox(legend_box, filled=False)
                )
            self.assertEqual(len(fig.axes), 1)
            self.assertFalse(ax.texts)
            self.assertAlmostEqual(fig.get_figwidth() / fig.get_figheight(), 6.6 / 4.5)
        finally:
            plt.close(fig)

    def test_caption_is_document_text_outside_the_plot(self):
        from markdown_it import MarkdownIt
        from PIL import Image

        source = (HERE / "manuscript.md").read_text()
        tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(source)
        starts = [i for i, token in enumerate(tokens) if token.type == "table_open"]
        self.assertEqual(len(starts), 2)
        self.assertNotIn("<style", source)
        self.assertNotIn("background: #", source)
        self.assertNotIn("font:", source)
        self.assertFalse(any(token.type == "html_block" for token in tokens))
        self.assertNotIn("<table", source)
        self.assertNotIn("<img", source)
        stop = next(i for i in range(starts[1], len(tokens)) if tokens[i].type == "table_close")
        cells = [token for token in tokens[starts[1] : stop] if token.type == "inline"]
        self.assertEqual(len(cells), 4)
        self.assertFalse(cells[0].content or cells[1].content)
        self.assertTrue(
            any(
                child.type == "image"
                and child.attrGet("src") == "figures/observation_limits_preview.png"
                for child in cells[2].children or []
            )
        )
        caption = cells[3]
        self.assertTrue(caption.content.startswith("**Figure 2.**"))
        document = (HERE / "manuscript.html").read_text()

        class FigureParser(HTMLParser):
            picture = None
            caption = None
            words = ""
            in_caption = False
            table = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "img" and attrs.get("src") == "figures/observation_limits.png":
                    self.picture = attrs
                if tag == "table" and attrs.get("id") == "figure-2":
                    self.table = attrs
                if attrs.get("id") == "figure-2-caption":
                    self.caption = attrs
                    self.in_caption = True

            def handle_endtag(self, tag):
                if tag == "div":
                    self.in_caption = False

            def handle_data(self, data):
                if self.in_caption:
                    self.words += data

        figure = FigureParser()
        figure.feed(document)
        self.assertIn("Dotted line: 5% reference", figure.words)
        self.assertEqual(figure.picture["aria-describedby"], figure.caption["id"])
        self.assertIn("table-layout: fixed", figure.table["style"])
        self.assertEqual(figure.table["role"], "presentation")
        self.assertEqual((figure.picture["width"], figure.picture["height"]), ("528", "360"))
        self.assertIn("height: auto !important", figure.picture["style"])
        self.assertIn("max-width: 528px", figure.picture["style"])
        self.assertIn("aspect-ratio: 22 / 15", figure.picture["style"])
        self.assertEqual(figure.words.strip(), caption.content.replace("**", ""))
        with Image.open(HERE / "figures/observation_limits.png") as image:
            self.assertEqual(image.size, (1320, 900))
            self.assertNotIn("Description", image.info)
        with Image.open(HERE / "figures/observation_limits_preview.png") as image:
            self.assertEqual(image.size, (528, 360))
        svg = ElementTree.parse(HERE / "figures/observation_limits.svg")
        self.assertNotIn("Figure 2.", "".join(svg.getroot().itertext()))

    def test_manuscript_html_matches_source_and_renders_equations(self):
        import tempfile

        from render_manuscript import render

        with tempfile.TemporaryDirectory() as folder:
            generated = render(Path(folder) / "manuscript.html").read_text()
        self.assertEqual(generated, (HERE / "manuscript.html").read_text())
        source_hash = hashlib.sha256((HERE / "manuscript.md").read_bytes()).hexdigest()
        self.assertIn(f'name="source-sha256" content="{source_hash}"', generated)
        self.assertIn('<table id="figure-2" class="figure-table"', generated)
        self.assertIn("@media (max-width: 760px)", generated)
        self.assertIn("display: block; width: 100% !important", generated)
        self.assertIn("color-scheme: light dark", generated)
        self.assertIn("background: Canvas; color: CanvasText", generated)
        self.assertNotIn("background: #", generated)
        self.assertNotIn("#fafafa", generated)
        self.assertEqual(generated.count('<div class="equation">'), 4)
        self.assertEqual(generated.count('<math xmlns="http://www.w3.org/1998/Math/MathML"'), 4)
        self.assertNotIn("<script", generated)
        self.assertNotIn("figure-2-caption\n", generated)

    def test_all_figure_inputs_match_exported_hashes(self):
        manifest = json.loads((HERE / "figures/sources.json").read_text())
        self.assertEqual(len(manifest), 6)
        for path, expected in manifest.items():
            self.assertEqual(hashlib.sha256((HERE / path).read_bytes()).hexdigest(), expected)

    def test_exported_field_preserves_saved_harmonic_powers(self):
        frame = pd.read_csv(HERE / "figures/harmonic_field.csv")
        self.assertEqual(frame.shape, (401, 25))
        np.testing.assert_allclose(frame.elapsed_s, np.arange(401) * 0.1, atol=1e-12)
        x = frame.iloc[:, 1:].to_numpy()
        np.testing.assert_allclose(x.mean(axis=1), 0, atol=1e-14)
        expected = pd.read_csv(HERE / "results/harmonics/responses.csv")
        expected = expected[
            (expected.accel == 0.3)
            & (expected["dt"] == 0.01)
            & (expected.amplitude == 0.2)
            & (expected.window == "early")
        ]
        power = 2 * abs(np.fft.rfft(x, axis=1) / 24) ** 2
        for row in expected.itertuples():
            np.testing.assert_allclose(
                power[100:200, row.q].mean(), row.power, rtol=1e-9, atol=1e-16
            )

    def test_comparison_table_matches_conditional_rates(self):
        from markdown_it import MarkdownIt

        data = pd.read_csv(HERE / "results/uncertainty/physics_fit/diagnostics.csv")
        selected = data[(data.tau == 1) & (data.noise_index == 0)]
        tokens = (
            MarkdownIt("commonmark").enable("table").parse((HERE / "manuscript.md").read_text())
        )
        start = next(i for i, token in enumerate(tokens) if token.type == "table_open")
        stop = next(i for i in range(start, len(tokens)) if tokens[i].type == "table_close")
        cells = [token.content for token in tokens[start:stop] if token.type == "inline"][5:]
        table = [cells[i : i + 5] for i in range(0, len(cells), 5)]
        self.assertEqual(len(table), 6)
        methods = {"Known dynamics": "known", "TLS": "tls", "Restricted IDM": "restricted_idm"}
        cutoffs = {"Plug-in": "plugin", "Direct calibration": "direct_calibration"}
        for cells in table:
            group = selected[
                (selected.method == methods[cells[0]]) & (selected.calibration == cutoffs[cells[1]])
            ]
            for column, excitation in zip(cells[2:], ["direct_gap", "direct_speed", "harmonic_02"]):
                expected = group[group.excitation == excitation].rejection_fraction.mean() * 100
                self.assertLessEqual(abs(float(column.rstrip("%")) - expected), 0.00501)
        operators = pd.read_csv(HERE / "results/uncertainty/learned_operator/operators.csv")
        high = operators[operators.noise_index == 2]
        self.assertEqual(
            int(((high.method == "noise_corrected") & (high.status != "eligible")).sum()), 27
        )
        self.assertEqual(int(((high.method == "tls") & (high.status != "eligible")).sum()), 2)

    def test_figure_exports_and_scope(self):
        from PIL import Image

        text = (HERE / "manuscript.md").read_text()
        self.assertEqual(text.count("![") + text.count("<img "), 3)
        self.assertLess(len(text.split()), 3300)
        for name in ["harmonic_mechanism", "observation_limits", "empirical_baseline"]:
            with Image.open(HERE / "figures" / (name + ".png")) as image:
                rgb = np.asarray(image.convert("RGB"))
            self.assertGreater(rgb.shape[0], 600)
            self.assertGreater(np.mean(np.any(rgb < 230, axis=2)), 0.02)
            self.assertTrue((HERE / "figures" / (name + ".svg")).exists())
        self.assertIn("69,874 of 155,366", text)
        self.assertIn("prior information, parameter count, and fitting objective", text)
