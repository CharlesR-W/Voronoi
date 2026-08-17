"""Pure construction of the deterministic probe-bank artifact."""

from __future__ import annotations

import io
from collections.abc import Mapping

import numpy as np

from voronoi_lab.config import LabConfig
from voronoi_lab.core import JSONLike, canonical_json_bytes
from voronoi_lab.exp1.banks import (
    make_image_bootstrap_plan,
    make_probe_index_plan,
    make_site_bank,
)
from voronoi_lab.exp1.tracking2 import resolve_cut


class ProbeArtifactError(ValueError):
    """The signed config and input payload cannot define a valid probe bank."""


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def build_probe_bank_files(
    config: LabConfig, input_payload: Mapping[str, object]
) -> tuple[dict[str, bytes], dict[str, JSONLike]]:
    """Return every byte of the probe artifact from signed, portable inputs."""

    training = input_payload.get("training")
    if not isinstance(training, Mapping):
        raise ProbeArtifactError("Tracking2 input artifact is missing training metadata")
    train_size = training.get("train_size")
    test_size = training.get("test_size")
    if (
        isinstance(train_size, bool)
        or not isinstance(train_size, int)
        or isinstance(test_size, bool)
        or not isinstance(test_size, int)
        or train_size < 1
        or test_size < 1
    ):
        raise ProbeArtifactError("Tracking2 input artifact has invalid dataset sizes")
    banks = config.experiment1.probe_banks
    if not banks.equal_weight_per_image:
        raise ProbeArtifactError("the current probe implementation requires equal image weights")
    plan = make_probe_index_plan(
        train_size=train_size,
        test_size=test_size,
        fit_train_images=banks.fit_train_images,
        independent_fit_train_images=banks.independent_fit_train_images,
        geometry_test_images=banks.geometry_test_images,
        intervention_test_images=banks.intervention_test_images,
        intervention_nested_in_geometry=banks.intervention_nested_in_geometry,
        root_seed=config.protocol.root_seed,
    )
    root_seed = config.protocol.root_seed
    files: dict[str, bytes] = {}
    roles: dict[str, JSONLike] = {}
    for role, indices in sorted(plan.roles.items()):
        index_file = f"indices/{role}.npy"
        files[index_file] = _npy_bytes(indices.astype("<i8", copy=False))
        split = "train" if role in {"codebook_fit", "independent_codebook_fit"} else "test"
        site_files: dict[str, JSONLike] = {}
        for cut_name in config.experiment1.cuts:
            cut = resolve_cut(cut_name)
            site_bank = make_site_bank(
                indices,
                height=cut.height,
                width=cut.width,
                max_sites_per_image=banks.max_sites_per_image,
                root_seed=root_seed,
                namespace=f"{role}/{cut.name}",
            )
            coordinates = np.column_stack(
                (site_bank.image_ids, site_bank.rows, site_bank.columns)
            ).astype("<i4", copy=False)
            site_file = f"sites/{role}/{cut.name}.npy"
            files[site_file] = _npy_bytes(coordinates)
            site_files[cut.name] = {
                "activation_shape": list(cut.activation_shape),
                "count": len(coordinates),
                "file": site_file,
            }
        role_metadata: dict[str, JSONLike] = {
            "count": len(indices),
            "index_file": index_file,
            "site_files": site_files,
            "split": split,
        }
        if role in {"geometry", "intervention"}:
            bootstrap = make_image_bootstrap_plan(
                indices,
                resamples=config.experiment1.bootstrap.resamples,
                root_seed=root_seed,
                namespace=role,
            )
            bootstrap_file = f"bootstrap/{role}.npy"
            files[bootstrap_file] = _npy_bytes(bootstrap.astype("<i8", copy=False))
            role_metadata["bootstrap_file"] = bootstrap_file
            role_metadata["bootstrap_shape"] = list(bootstrap.shape)
        roles[role] = role_metadata

    plan_payload: dict[str, JSONLike] = {
        "schema_version": 1,
        "bootstrap": config.experiment1.bootstrap.model_dump(mode="json"),
        "cuts": list(config.experiment1.cuts),
        "equal_weight_per_image": True,
        "intervention_nested_in_geometry": banks.intervention_nested_in_geometry,
        "max_sites_per_image": banks.max_sites_per_image,
        "roles": roles,
        "root_seed": root_seed,
        "site_coordinate_columns": ["image_id", "row", "column"],
        "site_weight_rule": "equal total mass per image; uniform over saved sites per image",
    }
    files["plan.json"] = canonical_json_bytes(plan_payload)
    return files, plan_payload


__all__ = ["ProbeArtifactError", "build_probe_bank_files"]
