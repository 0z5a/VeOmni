"""CPU integration and source-provenance tests; no GPU kernels are executed."""

import ast
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.experiments import tilereuse
from scripts.experiments.tilereuse import veomni


ROOT = Path(__file__).resolve().parents[2]


class SourceProvenance(unittest.TestCase):
    def test_numerical_source_matches_recorded_extraction(self):
        manifest = json.loads((ROOT / "scripts/experiments/tilereuse/provenance.json").read_text())

        class WithoutImports(ast.NodeTransformer):
            def visit_Import(self, node):
                return None

            def visit_ImportFrom(self, node):
                return None

        for item in manifest["files"]:
            with self.subTest(source=item["source"]):
                target = (ROOT / item["target"]).read_bytes()
                self.assertEqual(hashlib.sha256(target).hexdigest(), item["target_sha256"])
                tree = WithoutImports().visit(ast.parse(target))
                for node in ast.walk(tree):
                    if hasattr(node, "type_params") and not node.type_params:
                        del node.type_params
                digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
                self.assertEqual(digest, item["numerical_ast_sha256"])

    def test_public_import_does_not_load_gpu_or_bench_modules(self):
        script = (
            "import sys; from scripts.experiments import tilereuse; "
            "assert not any(k.split('.')[0] in "
            "{'torch', 'triton', 'tilelang', 'veomni', 'bench'} for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)

    def test_configuration_matches_frozen_dispatch(self):
        manifest = json.loads((ROOT / "scripts/experiments/tilereuse/provenance.json").read_text())
        reference = manifest["dispatch_reference"]
        self.assertEqual(set(tilereuse.FIXED_CONFIGS), set(reference["fixed_ids"]))
        for frozen in reference["fixed_configs"]:
            current = tilereuse.FIXED_CONFIGS[frozen["id"]]
            self.assertEqual(
                (current.group, current.backward_block, current.backward_head_tile, current.pad_heads),
                (frozen["group"], frozen["block"], frozen["head_tile"], frozen["pad"]),
            )
        for heads in (8, 16, 32, 64, 128):
            expected = reference["shape_rule"].get(str(heads), reference["fallback"])
            self.assertEqual(tilereuse.select(heads), expected)


class VeOmniBinding(unittest.TestCase):
    def setUp(self):
        self.original = object()
        self.mask = object()
        self.native = SimpleNamespace(
            eager_attention_forward=self.original,
            mask_sparse_attention_indices=self.mask,
        )
        self.bindings = []
        guard = patch.object(veomni, "_require_gpu")
        guard.start()
        self.addCleanup(guard.stop)

    def fake_bind(self, proxy, mode, **options):
        self.assertIs(proxy.mask_sparse_attention_indices, self.mask)
        self.assertIs(self.native.eager_attention_forward, self.original)
        self.bindings.append((mode, options))

        def attention(*args, **kwargs):
            return mode, options, args, kwargs

        proxy.eager_attention_forward = attention

    def test_dispatch_uses_current_shape_and_preserves_arguments(self):
        with patch("scripts.experiments.tilereuse._adapter.bind_chunk_attention", side_effect=self.fake_bind):
            attention = tilereuse.make_attention(self.native, profile=True)
        self.assertIs(self.native.eager_attention_forward, self.original)
        self.assertEqual(len(self.bindings), 4)
        for heads, group in ((8, 8), (16, 4), (32, 2), (64, 1), (8, 8)):
            query = SimpleNamespace(shape=(1, heads, 17, 512))
            args = (object(), query, object(), object(), object(), 512**-0.5)
            result = attention(*args, sparse_topk_indices="current-ids")
            self.assertEqual(result[1]["group_size"], group)
            self.assertEqual(result[2], args)
            self.assertEqual(result[3], {"sparse_topk_indices": "current-ids"})
            self.assertTrue(result[1]["profile"])
            self.assertTrue(result[1]["elide_singleton_cat"])
            self.assertTrue(result[1]["elide_zero_padding"])
        self.assertEqual(sum(attention.choice_counts.values()), 5)
        self.assertEqual(attention.choice_counts["g8-b32-m64-both"], 2)

    def test_fixed_configuration_does_not_construct_other_paths(self):
        with patch("scripts.experiments.tilereuse._adapter.bind_chunk_attention", side_effect=self.fake_bind):
            tilereuse.make_attention(self.native, "g4-b32-m64-both")
        self.assertEqual(len(self.bindings), 1)
        mode, options = self.bindings[0]
        self.assertEqual(mode, "grouped_reuse")
        self.assertEqual(options["group_size"], 4)
        self.assertEqual(options["backward_block_size"], 32)
        self.assertEqual(options["backward_max_heads"], 64)
        self.assertFalse(options["pad_grouped_heads"])

    def test_context_restores_after_nested_exception(self):
        outer, inner = object(), object()
        with patch.object(veomni, "make_attention", side_effect=[outer, inner]):
            with tilereuse.attention_context(self.native) as installed:
                self.assertIs(installed, outer)
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    with tilereuse.attention_context(self.native):
                        self.assertIs(self.native.eager_attention_forward, inner)
                        raise RuntimeError("training failed")
                self.assertIs(self.native.eager_attention_forward, outer)
        self.assertIs(self.native.eager_attention_forward, self.original)

    def test_construction_failure_does_not_change_native(self):
        calls = [0]

        def fail_on_second(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise ImportError("missing kernel dependency")
            return object()

        with patch.object(veomni, "_fixed_attention", side_effect=fail_on_second):
            with self.assertRaisesRegex(ImportError, "missing kernel dependency"):
                with tilereuse.attention_context(self.native):
                    self.fail("body must not run")
        self.assertIs(self.native.eager_attention_forward, self.original)

    def test_unsupported_hardware_fails_before_kernel_construction(self):
        with patch.object(veomni, "_require_gpu", side_effect=RuntimeError("SM90 required")):
            with patch.object(veomni, "_fixed_attention") as fixed:
                with self.assertRaisesRegex(RuntimeError, "SM90 required"):
                    tilereuse.make_attention(self.native)
                fixed.assert_not_called()
        self.assertIs(self.native.eager_attention_forward, self.original)

    def test_unknown_configuration_fails_before_kernel_import(self):
        with patch.object(veomni, "_fixed_attention") as fixed:
            with self.assertRaisesRegex(ValueError, "Unknown TileReuse configuration"):
                tilereuse.make_attention(self.native, "unknown")
            fixed.assert_not_called()
        self.assertIs(self.native.eager_attention_forward, self.original)


if __name__ == "__main__":
    unittest.main()
