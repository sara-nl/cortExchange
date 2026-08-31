"""
Re-upload existing weights so they carry an architecture manifest.

Weights uploaded before manifests existed have no record of which architecture they belong to, so
loading them under the wrong one is silent. This walks a group, works out what each model is, and
re-uploads it unchanged apart from the added manifest.

The architecture is detected from the model class the checkpoint pickles, read without executing
it. That distinguishes the V1-era models from the astroNNomy-era ones, but not V2 from V3 - the
same checkpoint loads identically under both, they differ only in preprocessing - so
--legacy-architecture decides that, and --override handles individual models.

Dry run by default:

    python scripts/rearchive_weights.py --group surf
    python scripts/rearchive_weights.py --group surf --write
"""

import argparse
import hashlib
import io
import os
import pickletools
import sys
import zipfile

from cortexchange.wdclient import DefaultWebdavArgs, client, init_downloader

# Models whose architecture is known and is not the legacy default.
OVERRIDES = {
    "surf/dinov2_vitb14_frozen_O2_aug_0975": "surf/TransferLearningV3",
    "surf/dinov2_vitb14_lora_O2_aug_0984": "surf/TransferLearningV3",
    "surf/dinov3_vitb16_frozen_O2_aug_0972": "surf/TransferLearningV3",
    "surf/dinov3_vitb16_lora_O2_aug_0982": "surf/TransferLearningV3",
}

# Which architecture goes with the model class a checkpoint pickles. The astroNNomy entry is
# resolved at runtime from --legacy-architecture, since the checkpoint cannot distinguish V2 from V3.
V1_MARKERS = ("__main__", "train_nn")
ASTRONNOMY_MARKER = "astronnomy"


def checkpoint_file(path):
    """
    The .pth to inspect. Some models were uploaded as a directory of files rather than a bare
    checkpoint, which the loaders accept too.
    """
    if os.path.isfile(path):
        return path
    candidates = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(".pth")
    )
    return candidates[0] if len(candidates) == 1 else None


def pickled_modules(path):
    """
    Module names referenced by a torch checkpoint, read without unpickling it.

    Unpickling would import - and in the V1 case fail to import - the very classes we are trying
    to identify, so the pickle is walked as opcodes instead.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not names:
                return set()
            raw = z.read(names[0])
    except zipfile.BadZipFile:
        # torch's pre-zip format: fall back to scanning for readable module strings
        with open(path, "rb") as f:
            head = f.read(4_000_000)
        return {m.decode("ascii", "ignore") for m in head.split(b"\x00") if b"." in m and m.isascii()}

    found = set()
    for op, arg, _ in pickletools.genops(io.BytesIO(raw)):
        if isinstance(arg, str):
            found.add(arg)
    return found


def detect_architecture(path, legacy_architecture):
    checkpoint = checkpoint_file(path)
    if checkpoint is None:
        return None, "no single .pth"
    refs = pickled_modules(checkpoint)
    blob = " ".join(refs)
    if ASTRONNOMY_MARKER in blob:
        return legacy_architecture, "astronnomy-format"
    if any(marker in blob for marker in V1_MARKERS):
        return "surf/TransferLearning", "v1-format"
    return None, "unrecognised"


def sha256(path):
    """Content digest of a file, or of a whole directory tree in sorted order."""
    h = hashlib.sha256()

    def eat(p):
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

    if os.path.isfile(path):
        eat(path)
    else:
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                full = os.path.join(root, name)
                h.update(os.path.relpath(full, path).encode())
                eat(full)
    return h.hexdigest()


def model_names(group):
    entries = client.list_group(group, list_weights=True)
    return sorted(
        f"{group}/{e[:-len('.tar.gz')]}" for e in entries if e.endswith(".tar.gz")
    )


def main(args):
    init_downloader(
        url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache
    )

    names = model_names(args.group)
    if args.only:
        wanted = {n if "/" in n else f"{args.group}/{n}" for n in args.only}
        names = [n for n in names if n in wanted]
    print(f"{len(names)} model(s) in '{args.group}'\n")

    overrides = dict(OVERRIDES)
    for item in args.override:
        name, _, arch = item.partition("=")
        overrides[name if "/" in name else f"{args.group}/{name}"] = arch

    planned, skipped, done, failed = [], [], [], []
    for name in names:
        # force the download: a cache filled before manifests existed would look manifest-less
        # even when the remote copy already has one
        client.download_model(name, force=True)
        weights = client.local_weights_path(name)

        existing = client.read_model_manifest(name)
        if existing and not args.replace_existing:
            skipped.append((name, f"already records {existing.get('architecture')}"))
            continue

        architecture = overrides.get(name)
        source = "override"
        if architecture is None:
            architecture, source = detect_architecture(weights, args.legacy_architecture)
        if architecture is None:
            skipped.append((name, f"{source}: cannot tell which architecture"))
            continue

        planned.append((name, weights, architecture, source))
        print(f"  {name.split('/')[-1]:44s} {source:16s} -> {architecture}")

    if not args.write:
        print(f"\n{len(planned)} to re-upload, {len(skipped)} skipped. Re-run with --write.")
    else:
        print()
        for name, weights, architecture, _ in planned:
            before = sha256(weights)
            try:
                client.upload_model(name, weights, architecture=architecture, force=True)
            except Exception as error:  # noqa: BLE001 - report and continue with the rest
                failed.append((name, f"upload failed: {type(error).__name__}: {error}"))
                print(f"  FAILED  {name}: {error}")
                continue

            client.download_model(name, force=True)
            manifest = client.read_model_manifest(name) or {}
            if sha256(client.local_weights_path(name)) != before:
                failed.append((name, "weights differ after round trip"))
                print(f"  FAILED  {name}: weights differ after round trip")
            elif manifest.get("architecture") != architecture:
                failed.append((name, f"manifest reads {manifest.get('architecture')!r}"))
                print(f"  FAILED  {name}: manifest reads {manifest.get('architecture')!r}")
            else:
                done.append(name)
                print(f"  ok      {name.split('/')[-1]:44s} {architecture}")
        print(f"\n{len(done)} re-uploaded, {len(skipped)} skipped, {len(failed)} failed.")

    for name, why in skipped:
        print(f"  skipped {name.split('/')[-1]:44s} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default="surf")
    parser.add_argument(
        "--legacy-architecture",
        default="surf/TransferLearningV2",
        help="Architecture to record for astroNNomy-format checkpoints, which cannot themselves "
             "distinguish V2 from V3 (default: %(default)s)",
    )
    parser.add_argument("--override", action="append", default=[], metavar="NAME=ARCHITECTURE")
    parser.add_argument("--only", nargs="+", default=[], help="Restrict to these model names")
    parser.add_argument("--replace-existing", action="store_true", help="Also rewrite models that already have a manifest")
    parser.add_argument("--write", action="store_true", help="Actually re-upload (default is a dry run)")
    parser.add_argument("--wd-url", default=DefaultWebdavArgs.URL)
    parser.add_argument("--wd-login", default=DefaultWebdavArgs.LOGIN)
    parser.add_argument("--wd-password", default=DefaultWebdavArgs.PASSWORD)
    parser.add_argument("--cache", default=DefaultWebdavArgs.CACHE)
    sys.exit(main(parser.parse_args()))
