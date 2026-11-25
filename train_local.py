# ============================================================
#  Banana AI v2 - เทรนในเครื่อง (RTX 5060)
#  ขั้นที่ 1: YOLOv8 หาตำแหน่งกล้วย
#  ขั้นที่ 2: HSV + Logistic Regression บอกระดับความสุก
#  วิธีรัน:  cd /d "D:\Banana Ai"   แล้ว   py train_local.py
# ============================================================
import os, sys, glob, json, shutil
sys.stdout.reconfigure(errors="replace")

BASE        = r"D:\Banana Ai"
YOLO_ROOT   = os.path.join(BASE, "banana_data", "data_banana")
WHITE_ROOT  = os.path.join(BASE, "Banana Ripeness Detection Dataset", "Banana Ripeness Detection Dataset")
DS          = os.path.join(BASE, "yolo_ds")        # dataset ที่จัดใหม่สำหรับ YOLO
OUT         = os.path.join(BASE, "web")            # ไฟล์สำหรับเว็บ
RESULTS     = os.path.join(BASE, "results")        # ภาพผลลัพธ์ตัวอย่าง
MY_TESTS    = os.path.join(BASE, "my_tests")       # (ไม่บังคับ) ใส่รูปของตัวเองไว้ทดสอบ
EPOCHS      = 30
EXPORT_SIZE = 320
SEED        = 42

import numpy as np, cv2
IMG_SIZE = 256
CLASS_NAMES = ["Green", "Semi-ripe", "Ripe", "Overripe"]
COLORS = {"Green": (58,158,90), "Semi-ripe": (46,194,181), "Ripe": (24,197,245), "Overripe": (43,74,107)}

def banana_mask(hsv):
    h, s, v = cv2.split(hsv)
    mask = ((s > 50) | (v < 90)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = banana_mask(hsv)
    px = hsv[m > 0].astype(float)
    if len(px) < 100:
        px = hsv.reshape(-1, 3).astype(float)
    return np.concatenate([px.mean(axis=0), px.std(axis=0)])

def crop_box(img, cx, cy, w, h, pad=0.05):
    H, W = img.shape[:2]
    x1 = int(max(0, (cx - w/2 - pad*w) * W)); x2 = int(min(W, (cx + w/2 + pad*w) * W))
    y1 = int(max(0, (cy - h/2 - pad*h) * H)); y2 = int(min(H, (cy + h/2 + pad*h) * H))
    return img[y1:y2, x1:x2]

def step(t): print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60, flush=True)

def main():
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    step(f"อุปกรณ์ที่ใช้เทรน: {torch.cuda.get_device_name(0) if device == 0 else 'CPU (ไม่เจอ GPU!)'}")
    assert os.path.isdir(YOLO_ROOT), f"ไม่เจอ {YOLO_ROOT}"
    for d in (OUT, RESULTS): os.makedirs(d, exist_ok=True)

    # ---------- 1) เตรียม dataset YOLO (รวมทุกกรอบเป็นคลาส banana) ----------
    step("1) เตรียม dataset สำหรับ YOLO")
    shutil.rmtree(DS, ignore_errors=True)
    splits = []
    for split in ["train", "val", "test"]:
        src = os.path.join(YOLO_ROOT, split)
        if not os.path.isdir(os.path.join(src, "images")): continue
        os.makedirs(os.path.join(DS, split, "images")); os.makedirs(os.path.join(DS, split, "labels"))
        n = 0
        for img in glob.glob(os.path.join(src, "images", "*")):
            name = os.path.splitext(os.path.basename(img))[0]
            shutil.copy(img, os.path.join(DS, split, "images"))
            lab = os.path.join(src, "labels", name + ".txt")
            lines = open(lab).read().splitlines() if os.path.exists(lab) else []
            lines = ["0 " + l.split(maxsplit=1)[1] for l in lines if len(l.split()) == 5]
            open(os.path.join(DS, split, "labels", name + ".txt"), "w").write("\n".join(lines))
            n += 1
        splits.append(split); print(f"   {split:5s}: {n} รูป")
    val_split = "val" if "val" in splits else "test"
    test_split = "test" if "test" in splits else val_split
    yaml_path = os.path.join(DS, "data.yaml")
    open(yaml_path, "w").write(
        f"path: {DS.replace(os.sep, '/')}\ntrain: train/images\nval: {val_split}/images\n"
        f"test: {test_split}/images\nnames:\n  0: banana\n")

    # ---------- 2) เทรน YOLO ----------
    step(f"2) เทรน YOLOv8n {EPOCHS} รอบ")
    yolo = YOLO("yolov8n.pt")
    yolo.train(data=yaml_path, epochs=EPOCHS, imgsz=640, batch=16, device=device, workers=2,
               seed=SEED, project=os.path.join(BASE, "runs"), name="banana", exist_ok=True)
    best = os.path.join(BASE, "runs", "banana", "weights", "best.pt")
    yolo = YOLO(best)
    m = yolo.val(data=yaml_path, split=test_split, device=device)
    yolo_map50, yolo_map = float(m.box.map50), float(m.box.map)

    # ---------- 3) เทรนโมเดลความสุก ----------
    step("3) เทรนโมเดลความสุก (HSV + Logistic Regression)")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    X, y, src = [], [], []
    if os.path.isdir(WHITE_ROOT):
        for i, c in enumerate(CLASS_NAMES):
            for f in glob.glob(os.path.join(WHITE_ROOT, c, "*")):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(f)
                    if img is not None: X.append(extract_features(img)); y.append(i); src.append("white")
    MAP = {0: 0, 1: 2}   # Raw -> Green, Ripe -> Ripe
    for split in splits:
        for lab in glob.glob(os.path.join(YOLO_ROOT, split, "labels", "*.txt")):
            name = os.path.splitext(os.path.basename(lab))[0]
            imgs = glob.glob(os.path.join(YOLO_ROOT, split, "images", name + ".*"))
            if not imgs: continue
            img = cv2.imread(imgs[0])
            if img is None: continue
            for line in open(lab):
                p = line.split()
                if len(p) != 5 or int(p[0]) not in MAP: continue
                crop = crop_box(img, *map(float, p[1:]))
                if crop.size == 0 or min(crop.shape[:2]) < 20: continue
                X.append(extract_features(crop)); y.append(MAP[int(p[0])]); src.append("real")
    X, y, src = np.array(X), np.array(y), np.array(src)
    for s in ["white", "real"]:
        print(f"   {s:5s}:", {CLASS_NAMES[i]: int(((y == i) & (src == s)).sum()) for i in range(4)})
    strat = np.char.add(src, y.astype(str))
    Xtr, Xte, ytr, yte, _, ste = train_test_split(X, y, src, test_size=0.2, stratify=strat, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, ytr); pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    accs = {s: accuracy_score(yte[ste == s], pred[ste == s]) for s in ["white", "real"] if (ste == s).any()}
    labs = sorted(set(yte))
    print(classification_report(yte, pred, labels=labs, target_names=[CLASS_NAMES[i] for i in labs], digits=3))
    print("Confusion matrix (แถว = จริง, คอลัมน์ = ทาย):\n", confusion_matrix(yte, pred, labels=labs))

    # ---------- 4) ทดสอบระบบเต็ม แล้วบันทึกภาพ ----------
    step("4) ทดสอบระบบเต็ม (YOLO -> HSV/LR)")
    def run_pipeline(img, conf=0.35):
        out = img.copy(); res = []
        for b in yolo.predict(img, conf=conf, verbose=False, device=device)[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            prob = clf.predict_proba([extract_features(crop)])[0]; name = CLASS_NAMES[prob.argmax()]
            res.append(f"{name} {prob.max()*100:.0f}%")
            cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[name], 3)
            cv2.putText(out, f"{name} {prob.max()*100:.0f}%", (x1 + 4, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[name], 2)
        return out, res
    tests = sorted(glob.glob(os.path.join(DS, test_split, "images", "*")))
    tests = tests[::max(1, len(tests) // 8)][:8]
    if os.path.isdir(MY_TESTS): tests += glob.glob(os.path.join(MY_TESTS, "*"))
    for f in tests:
        img = cv2.imread(f)
        if img is None: continue
        out, res = run_pipeline(img)
        cv2.imwrite(os.path.join(RESULTS, "result_" + os.path.basename(f)), out)
        print(f"   {os.path.basename(f)[:40]:40s} -> {res if res else 'ไม่พบกล้วย'}")

    # ---------- 5) Export สำหรับเว็บ ----------
    step("5) Export ไฟล์สำหรับเว็บ")
    onnx = YOLO(best).export(format="onnx", imgsz=EXPORT_SIZE, opset=12, simplify=True)
    shutil.copy(onnx, os.path.join(OUT, "banana_yolo.onnx"))
    sc = clf.named_steps["standardscaler"]; lr = clf.named_steps["logisticregression"]
    W = {"classes": CLASS_NAMES, "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
         "coef": lr.coef_.round(6).tolist(), "intercept": lr.intercept_.round(6).tolist(),
         "yolo_size": EXPORT_SIZE}
    json.dump(W, open(os.path.join(OUT, "ripeness_weights.json"), "w"))

    step("เสร็จแล้ว! สรุปผล")
    print(f"   YOLO  mAP50 = {yolo_map50:.3f}   mAP50-95 = {yolo_map:.3f}")
    print(f"   Ripeness accuracy (รวม) = {acc*100:.2f}%   " + "   ".join(f"{k} = {v*100:.2f}%" for k, v in accs.items()))
    print(f"   ไฟล์เว็บ  : {OUT}")
    print(f"   ภาพผลลัพธ์: {RESULTS}")
    print(f"   กราฟเทรน  : {os.path.join(BASE, 'runs', 'banana')}")

if __name__ == "__main__":
    main()
