#!/usr/bin/env python3
"""One repository-owned way to name a local model in output.

A model's identity here is the weight file the runtime actually opens, so every
display starts from the GGUF filename on disk. The catalog's ``source`` is a
different fact: the publisher's name for the release those bytes came from. The
``heretic`` preset is why the two are kept apart. Its artifact is
``RVN-Q6_K-multilingual-mtp.gguf`` while its upstream release is published as
``Qwen3.8-27B-Heretic-Abliterated-Uncensored``, so a listing that prints only
the source names a file that exists nowhere on this disk, and one that prints
only the alias names nothing an operator can check against ``llama-models.ini``.

Aliases still work everywhere they worked before -- commands, the API and the
INI all still take ``heretic`` -- but they no longer lead a display. A short
handle is a compatibility affordance, not an identity: it tells an operator
nothing they can check against the disk, and two different weight files could
wear the same one. Displays therefore read
``<artifact> (<intended use>)  [compatibility alias: <handle>]``.
"""
from __future__ import annotations

import configparser
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "llama-models.ini"
CATALOG = ROOT / "config/model-catalog.json"

# Source is provenance, purpose is intended use. Neither is an identity, and
# neither may be absent: a preset with a blank purpose would print as a bare
# filename and read as a display bug rather than as missing catalog data.
CATALOG_FIELDS = frozenset({"source", "purpose"})

UNCATALOGUED = "no catalog entry"
UNCONFIGURED = "not a configured preset"
# Short handles like "heretic" are kept because commands, the API and existing
# scripts pass them, but they identify nothing on disk. Every display leads with
# the engineering identity and labels the handle for what it is.
ALIAS_LABEL = "compatibility alias"


def presets(path=PRESETS) -> dict[str, dict[str, str]]:
    """Router presets, with the shared ``[*]`` section merged into every alias."""
    ini = configparser.ConfigParser(interpolation=None)
    with Path(path).open() as handle:
        ini.read_file(handle)
    common = dict(ini["*"]) if "*" in ini else {}
    return {name: {**common, **dict(ini[name])}
            for name in ini.sections() if name != "*"}


def load_catalog(path=CATALOG) -> dict[str, dict[str, str]]:
    """Load the catalog or raise; every entry must carry both fields, non-empty."""
    catalog = json.loads(Path(path).read_text())
    if not isinstance(catalog, dict):
        raise ValueError("model catalog must be a JSON object")
    for alias, row in catalog.items():
        if (not isinstance(row, dict) or set(row) != CATALOG_FIELDS
                or not all(isinstance(value, str) and value.strip()
                           for value in row.values())):
            raise ValueError(f"invalid catalog entry: {alias}")
    return catalog


def artifacts(preset: dict) -> list[str]:
    """The weight filenames a preset loads, in the order llama-server takes them."""
    names = [Path(preset["model"]).name] if preset.get("model") else []
    if preset.get("mmproj"):
        names.append(Path(preset["mmproj"]).name)
    return names


def identity(preset: dict) -> str:
    """The full artifact filenames, projector included.

    A vision preset shares its weight file with the text-only sibling it was
    derived from, so the projector is the only thing that tells the two listing
    lines apart. Naming it is the difference between a duplicate-looking entry
    and an accurate one.
    """
    return " + ".join(artifacts(preset)) or UNCONFIGURED


def purpose(alias: str, catalog: dict) -> str:
    row = catalog.get(alias)
    return row["purpose"] if row else UNCATALOGUED


def describe(alias: str, preset: dict, catalog: dict) -> str:
    """``<artifact filenames> (<intended use>)`` for one configured preset."""
    return f"{identity(preset)} ({purpose(alias, catalog)})"


def describe_alias(alias: str, registry: dict, catalog: dict) -> str:
    """``describe`` for a caller holding only an alias, such as a router model id.

    An id the router publishes but this checkout does not configure is a fact
    worth printing plainly, not a reason to abort a read-only status summary.
    That is the only case where the handle is all there is to print.
    """
    preset = registry.get(alias)
    if preset is None:
        return f"{alias} ({UNCONFIGURED})"
    return describe(alias, preset, catalog)


def labelled(alias: str, registry: dict, catalog: dict) -> str:
    """Identity first, compatibility handle tagged behind it."""
    if alias not in registry:
        return describe_alias(alias, registry, catalog)
    return f"{describe_alias(alias, registry, catalog)}  [{ALIAS_LABEL}: {alias}]"


def listing_line(alias: str, preset: dict, catalog: dict, width: int = 0) -> str:
    """One ``artifacts (intended use)  [compatibility alias: x]`` row.

    The engineering identity leads. An alias like ``heretic`` names nothing an
    operator can check against the disk, so it is demoted to what it actually
    is: a short compatibility handle the command line and the API still accept.
    ``width`` pads the identity so the alias tags line up.
    """
    body = f"{describe(alias, preset, catalog):<{width}}"
    return f"{body}  [{ALIAS_LABEL}: {alias}]".rstrip()


def describe_width(registry: dict, catalog: dict) -> int:
    """Column width for the identity field of a whole listing."""
    return max((len(describe(alias, preset, catalog))
                for alias, preset in registry.items()), default=0)


def local_registry() -> tuple[dict, dict]:
    """Presets and catalog as far as this checkout can read them, or empty.

    Reporting on a live router must not depend on local configuration files
    being present and well-formed. The paths that *must* fail on a broken
    catalog -- the serving listing and its tests -- call the strict loaders.
    """
    try:
        registry = presets(PRESETS)
    except (OSError, ValueError, configparser.Error):
        registry = {}
    try:
        catalog = load_catalog(CATALOG)
    except (OSError, ValueError):
        catalog = {}
    return registry, catalog
