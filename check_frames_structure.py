import os

FRAME_ROOT = "data/frames_thermal"

for split in ["train", "val", "test"]:
    split_path = os.path.join(FRAME_ROOT, split)
    print("\nSPLIT:", split, "exists:", os.path.isdir(split_path))

    if not os.path.isdir(split_path):
        continue

    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    print("  #classes:", len(classes))
    print("  sample classes:", classes[:5])

    if classes:
        c0 = classes[0]
        c0_path = os.path.join(split_path, c0)
        vids = [d for d in os.listdir(c0_path) if os.path.isdir(os.path.join(c0_path, d))]
        print("  sample class:", c0, " #video_folders:", len(vids))
        if vids:
            v0_path = os.path.join(c0_path, vids[0])
            imgs = [f for f in os.listdir(v0_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
            print("  sample video folder:", vids[0], " #frames:", len(imgs))
