"""
Group-aware train/val/test split for UCF101 (and an audit for existing splits).

Why: UCF101 clips are named v_<Class>_g<GG>_c<CC>.avi. All clips that share a
group number (g01..g25) come from the same long source video -- same actor,
same background, same camera. If clips from one group land in both train and
test, the model can recognize the *scene* instead of the *action*, and test
accuracy is inflated. The official UCF101 splits therefore split by group.

This script:
  * split mode (default): reads a UCF101-style source tree
        <src>/<Class>/v_<Class>_gXX_cYY.avi
    and builds data/raw_videos/{train,val,test}/<Class>/ by GROUP, so no group
    ever appears in two splits. Defaults follow official split 1
    (groups 1-7 = test) and hold out groups 8-10 for validation.
  * --audit: checks an existing data/raw_videos (or data/frames_thermal) tree
    and reports any (class, group) that appears in more than one split.

Examples:
    python 02_split_ucf101.py --src path/to/UCF101 --classes JumpingJack PushUps TaiChi
    python 02_split_ucf101.py --src data/raw_videos_old --mode copy   # re-split an existing split tree
    python 02_split_ucf101.py --audit
    python 02_split_ucf101.py --audit --root data/frames_thermal
"""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

SPLITS = ("train", "val", "test")
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")
GROUP_RE = re.compile(r"_g(\d+)_c\d+", re.IGNORECASE)


def parse_group(name):
    """Return the UCF101 group number from a clip/folder name, or None."""
    m = GROUP_RE.search(name)
    return int(m.group(1)) if m else None


def parse_groups(spec):
    """'1-7' or '1,2,3' or '1-3,9' -> {1,2,3,...}"""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# -----------------------------
# AUDIT
# -----------------------------
def audit(root):
    """Report (class, group) pairs that occur in more than one split."""
    seen = defaultdict(lambda: defaultdict(int))   # (cls, group) -> {split: n}
    unparsed = 0
    total = 0

    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"  (missing split folder: {split_dir})")
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for item in os.listdir(cls_dir):
                total += 1
                g = parse_group(item)
                if g is None:
                    unparsed += 1
                    continue
                seen[(cls, g)][split] += 1

    leaks = {k: v for k, v in seen.items() if len(v) > 1}

    print(f"Audited {total} clips under {root}")
    if unparsed:
        print(f"  WARNING: {unparsed} clip names had no _gXX_cYY group tag; they could not be checked.")

    if not leaks:
        print("  OK: no UCF101 group appears in more than one split.")
        return 0

    n_leaky_clips = sum(sum(v.values()) for v in leaks.values())
    print(f"  LEAKAGE: {len(leaks)} (class, group) pairs span multiple splits "
          f"({n_leaky_clips} clips affected).")
    for (cls, g), splits in sorted(leaks.items())[:25]:
        detail = ", ".join(f"{s}={n}" for s, n in splits.items())
        print(f"    {cls} g{g:02d}: {detail}")
    if len(leaks) > 25:
        print(f"    ... and {len(leaks) - 25} more")
    print("  Test accuracy on this split is likely optimistic. Re-split with this script.")
    return 1


# -----------------------------
# SPLIT
# -----------------------------
def place(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:  # hardlink
        os.link(src, dst)


def split(args):
    test_groups = parse_groups(args.test_groups)
    val_groups = parse_groups(args.val_groups)
    overlap = test_groups & val_groups
    if overlap:
        sys.exit(f"Groups {sorted(overlap)} are in both --test-groups and --val-groups.")

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        sys.exit("--src and --out must be different folders (rename the old split first).")
    if not os.path.isdir(args.src):
        sys.exit(f"Source folder not found: {args.src}")

    for s in SPLITS:
        d = os.path.join(args.out, s)
        if os.path.isdir(d) and os.listdir(d) and not args.force:
            sys.exit(f"{d} already has content. Move it away or pass --force "
                     f"(existing files are kept; new ones are added).")

    # Source can be a plain UCF101 tree (<src>/<Class>/*.avi) or an existing
    # split tree (<src>/{train,val,test}/<Class>/*.avi) whose clips get pooled.
    pooled = [os.path.join(args.src, s) for s in SPLITS if os.path.isdir(os.path.join(args.src, s))]
    roots = pooled or [args.src]
    if pooled:
        print(f"Source is an existing split tree; pooling clips from: {', '.join(os.path.basename(r) for r in pooled)}\n")

    clips = defaultdict(dict)   # cls -> {filename: path}
    for r in roots:
        for cls in os.listdir(r):
            cls_dir = os.path.join(r, cls)
            if not os.path.isdir(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                if vid.lower().endswith(VIDEO_EXTS):
                    clips[cls].setdefault(vid, os.path.join(cls_dir, vid))

    classes = args.classes or sorted(clips)

    counts = {s: 0 for s in SPLITS}
    for cls in classes:
        if cls not in clips:
            print(f"WARNING: class not found in source, skipping: {cls}")
            continue

        per_cls = {s: 0 for s in SPLITS}
        for vid, path in sorted(clips[cls].items()):
            g = parse_group(vid)
            if g is None:
                print(f"WARNING: no group tag in {vid}, skipping")
                continue
            s = "test" if g in test_groups else "val" if g in val_groups else "train"
            place(path, os.path.join(args.out, s, cls, vid), args.mode)
            per_cls[s] += 1

        for s in SPLITS:
            counts[s] += per_cls[s]
        print(f"{cls:25s} train={per_cls['train']:4d} val={per_cls['val']:4d} test={per_cls['test']:4d}")

    print(f"\nTOTAL train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"Test groups: {sorted(test_groups)}  Val groups: {sorted(val_groups)}")
    print(f"Written to: {args.out}\n")
    audit(args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", help="UCF101 root (one folder per class) or an existing train/val/test tree to re-split")
    p.add_argument("--out", default="data/raw_videos", help="Output root for train/val/test")
    p.add_argument("--classes", nargs="*", help="Class folder names to include (default: all)")
    p.add_argument("--test-groups", default="1-7", help="Groups for test (default 1-7 = official split 1)")
    p.add_argument("--val-groups", default="8-10", help="Groups for validation (default 8-10)")
    p.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink",
                   help="How to place videos (default symlink; use copy on Windows without symlink rights)")
    p.add_argument("--force", action="store_true", help="Allow writing into non-empty split folders")
    p.add_argument("--audit", action="store_true", help="Only check an existing split for group leakage")
    p.add_argument("--root", default="data/raw_videos", help="Tree to audit (with --audit)")
    args = p.parse_args()

    if args.audit:
        sys.exit(audit(args.root))
    if not args.src:
        p.error("--src is required unless --audit is given")
    split(args)


if __name__ == "__main__":
    main()
