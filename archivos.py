# -*- coding: utf-8 -*-
import re
import fitz
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import pytesseract

PDF_PATH = r"./pdfs/1943853451940.pdf"
OUT_CSV  = r"./certificados.csv"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PAGE_ZOOM = 5.0
LANG = "spa+eng"
FIELDS = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]

BOX_MAIN = (0.37, 0.28, 0.95, 0.52)
CERT_BOXES = [
    (0.62, 0.235, 0.97, 0.32),
    (0.55, 0.22,  0.98, 0.33),
    (0.60, 0.22,  0.98, 0.35),
    (0.50, 0.21,  0.98, 0.36),
]

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

# -------------------------
# Utils
# -------------------------
def norm_spaces(s: str) -> str:
    return " ".join((s or "").split())

def render_page(doc, page_index: int, zoom=PAGE_ZOOM) -> Image.Image:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def crop_rel(img: Image.Image, box):
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))

def crop_rel_inside(parent: Image.Image, sub_box):
    """sub_box in 0..1 coords inside parent"""
    w, h = parent.size
    x1, y1, x2, y2 = sub_box
    return parent.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))

# -------------------------
# Preprocess
# -------------------------
def preprocess_remove_lines(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    inv = 255 - th

    horiz = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (140, 1)), 1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (140, 1)), 1)

    vert = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 140)), 1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 140)), 1)

    lines = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.subtract(inv, lines)
    out = 255 - cleaned
    return Image.fromarray(out)

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    g = pil_img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=2)
    g = g.filter(ImageFilter.SHARPEN)
    return g

def preprocess_small_line(pil_img: Image.Image, scale=6.0) -> Image.Image:
    g = pil_img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=0)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=2))
    arr = np.array(g)
    arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(arr)

# -------------------------
# OCR
# -------------------------
def ocr_block(img: Image.Image, psm=6) -> str:
    cfg = f"--oem 1 --psm {psm}"
    return pytesseract.image_to_string(img, lang=LANG, config=cfg).replace("\x0c","").strip()

def ocr_line(img: Image.Image, whitelist: str, psm=7) -> str:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, lang="eng", config=cfg).replace("\x0c","").strip()

def ocr_data(img: Image.Image):
    df = pytesseract.image_to_data(img, lang=LANG, config="--oem 1 --psm 6",
                                   output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["u"] = df["text"].astype(str).str.upper()
    return df

# -------------------------
# Grid detection
# -------------------------
def cluster_positions(pos, gap=2):
    pos = sorted(pos)
    if not pos:
        return []
    clusters = []
    cur = [pos[0]]
    prev = pos[0]
    for p in pos[1:]:
        if p - prev <= gap:
            cur.append(p)
        else:
            clusters.append(int(np.median(cur)))
            cur = [p]
        prev = p
    clusters.append(int(np.median(cur)))
    return clusters

def detect_v_lines(main_pil: Image.Image):
    gray = np.array(main_pil.convert("L"))
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    inv = 255 - th
    vert = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT,(1,80)), 1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT,(1,80)), 1)
    vsum = (vert > 0).sum(axis=0)
    idx = np.where(vsum > main_pil.size[1]*0.2)[0]
    lines = cluster_positions(idx, 2)
    if len(lines) >= 3:
        lines = sorted(lines, key=lambda x: vsum[x], reverse=True)[:3]
        lines = sorted(lines)
    return lines

def detect_h_lines(main_pil: Image.Image):
    gray = np.array(main_pil.convert("L"))
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    inv = 255 - th
    horiz = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT,(120,1)), 1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT,(120,1)), 1)
    hsum = (horiz > 0).sum(axis=1)
    idx = np.where(hsum > main_pil.size[0]*0.3)[0]
    return cluster_positions(idx, 2)

def find_vehicle_rows(main_pil: Image.Image, v_lines, h_lines):
    if len(v_lines) < 3 or len(h_lines) < 6:
        return None
    left, mid = v_lines[0], v_lines[1]

    def score_block(ys):
        keys = ["MAR", "LIN", "MOD", "PLA", "COL"]
        score = 0
        for i in range(5):
            band = main_pil.crop((left, ys[i], mid, ys[i+1]))
            txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(band)), psm=6).upper()
            if keys[i] in txt:
                score += 3
        return score

    best = None
    for j in range(len(h_lines)-5):
        ys = h_lines[j:j+6]
        sc = score_block(ys)
        if best is None or sc > best[0]:
            best = (sc, ys)
    return best[1] if best and best[0] >= 3 else None

# -------------------------
# Normalizers + validators
# -------------------------
def norm_isuzu(txt: str) -> str:
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())
    if "ISUZU" in t or t in ("TSUZU","RSUZU","IISUZU","SUZU","TSU"):
        return "ISUZU"
    return ""

def fix_vin_ocr(txt: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (txt or "").upper())
    t = t.replace("JAAMPR","JAANPR")
    t = t.replace("I","1").replace("O","0").replace("Q","0")
    idx = t.find("JAANPR")
    if idx != -1 and idx + 17 <= len(t):
        t = t[idx:idx+17]
    if len(t) != 17 or not t.startswith("JAANPR"):
        return ""
    core = list(t)
    core[6], core[7] = "7","1"  # fuerza 71
    t = "".join(core)
    t = re.sub(r"HR110", "HR710", t)
    t = re.sub(r"HS110", "HS710", t)
    if len(t) == 17 and all(c in VIN_ALLOWED for c in t) and t.startswith("JAANPR"):
        return t
    return ""

def norm_np(txt: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (txt or "").upper())
    if "NP" in t or t in ("KP","PP","NIP"):
        return "NP"
    return ""

def norm_pas(txt: str) -> str:
    m = re.search(r"(\d{1,2})", txt or "")
    return m.group(1) if m else ""

def norm_year(txt: str) -> str:
    m = re.search(r"(19|20)\d{2}", txt or "")
    return m.group(0) if m else ""

def norm_camion(txt: str) -> str:
    t = (txt or "").upper()
    if "CAMION" in t or any(k in t for k in ["CANON","CAMON","CANNON","CAAION","CAAVON","CAAIION","CAAIOH"]):
        return "CAMION"
    return ""

def norm_blanco(txt: str) -> str:
    return "BLANCO" if "BLANCO" in (txt or "").upper() else ""

def norm_motor(raw: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    t = t.translate(str.maketrans({"O":"0","Q":"0","S":"5","I":"1","L":"1"}))
    # FIX REAL: M0XXXX -> 0XXXX (no perder el 0)
    t = re.sub(r"^M0", "0", t)
    t = re.sub(r"^M", "", t)
    t = re.sub(r"^N0V", "0V", t)
    if len(t) > 8:
        t = t[:8]
    return t if 3 <= len(t) <= 10 else ""

def norm_placa(raw: str) -> str:
    t = (raw or "").upper().replace(" ", "").replace(".", "-")
    t = re.sub(r"[^A-Z0-9\-]", "", t)
    m = re.search(r"([A-Z])\-?([A-Z0-9]{5,9})", t)
    if not m:
        return ""
    pref, body = m.group(1), m.group(2)
    trans = str.maketrans({"G":"6","O":"0","Q":"0","D":"0","B":"8","S":"5","Z":"2","I":"1","L":"1"})
    cand4 = re.sub(r"[^0-9]", "", body[:4].translate(trans))
    if len(cand4) == 4:
        letters = re.sub(r"[^A-Z]", "", body[4:])
        if 2 <= len(letters) <= 3:
            return f"{pref}-{cand4}{letters}"
    return ""  # <- estricto a 4 dígitos

def v_cert(x):   return bool(re.fullmatch(r"\d{3,4}", x or ""))
def v_marca(x):  return (x or "") == "ISUZU"
def v_chasis(x): return bool(x) and len(x)==17 and x.startswith("JAANPR")
def v_linea(x):  return (x or "") == "NP"
def v_pas(x):    return bool(re.fullmatch(r"\d{1,2}", x or ""))
def v_modelo(x): return bool(re.fullmatch(r"(19|20)\d{2}", x or ""))
def v_clase(x):  return (x or "") == "CAMION"
def v_color(x):  return (x or "") == "BLANCO"
def v_placa(x):  return bool(re.fullmatch(r"[A-Z]-\d{4}[A-Z]{2,3}", (x or "").upper()))
def v_motor(x):  return bool(re.fullmatch(r"[A-Z0-9]{3,10}", (x or "").upper()))

# -------------------------
# Fallbacks for Marca/Chasis (label + fixed crops)
# -------------------------
def find_label_bbox(df, key):
    cand = df[df["u"].str.contains(key)]
    if cand.empty:
        return None
    cand = cand.sort_values(["conf","width"], ascending=[False, False]).iloc[0]
    l = int(cand.left); t = int(cand.top); r = int(cand.left + cand.width); b = int(cand.top + cand.height)
    return l,t,r,b

def crop_right_of_label(main_rgb, bbox, right_limit=None, pad=6):
    if bbox is None:
        return None
    l,t,r,b = bbox
    w,h = main_rgb.size
    x1 = min(w, r + pad)
    x2 = right_limit if right_limit is not None else w
    return main_rgb.crop((x1, max(0,t-pad), x2, min(h,b+pad)))

def fallback_marca_chasis_label(main_rgb):
    clean = preprocess_for_ocr(preprocess_remove_lines(main_rgb))
    df = ocr_data(clean)
    bb_marca = find_label_bbox(df, "MARCA")
    bb_chas  = find_label_bbox(df, "CHAS")
    marca_img = crop_right_of_label(main_rgb, bb_marca, right_limit=int(main_rgb.size[0]*0.55))
    chas_img  = crop_right_of_label(main_rgb, bb_chas,  right_limit=main_rgb.size[0])
    marca_txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(marca_img))) if marca_img else ""
    chas_txt  = ocr_block(preprocess_for_ocr(preprocess_remove_lines(chas_img)))  if chas_img else ""
    marca = norm_isuzu(marca_txt)
    chasis = fix_vin_ocr(chas_txt)
    if not v_marca(marca): marca = ""
    if not v_chasis(chasis): chasis = ""
    return marca, chasis

def fallback_marca_fixed(main_rgb):
    # recortes candidatos dentro de BOX_MAIN (columna izquierda fila 1 aprox)
    cands = [
        (0.02, 0.18, 0.48, 0.28),
        (0.02, 0.16, 0.50, 0.30),
        (0.00, 0.18, 0.55, 0.29),
    ]
    for b in cands:
        img = crop_rel_inside(main_rgb, b)
        txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(img))).upper()
        if "ISUZU" in txt or any(x in txt for x in ["TSUZU","RSUZU","IISUZU","SUZU","TSU"]):
            return "ISUZU"
    return ""

def fallback_chasis_fixed(main_rgb):
    # recortes candidatos dentro de BOX_MAIN (columna derecha fila 1 aprox)
    cands = [
        (0.50, 0.18, 0.98, 0.28),
        (0.48, 0.16, 0.98, 0.30),
    ]
    for b in cands:
        img = crop_rel_inside(main_rgb, b)
        txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(img))).upper()
        vin = fix_vin_ocr(txt)
        if vin:
            return vin
    return ""

# -------------------------
# Cert extract
# -------------------------
def extract_cert(page_img: Image.Image) -> str:
    w, h = page_img.size
    for box in CERT_BOXES:
        crop = page_img.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
        clean = preprocess_for_ocr(preprocess_remove_lines(crop))
        txt = ocr_block(clean, psm=6).upper()
        m = re.search(r"CERT\w*\s*NO\.?\s*[:\.]?\s*(\d{3,4})", txt)
        if m:
            return m.group(1)
    return ""

# -------------------------
# Extract vehicle table (grid primary)
# -------------------------
def extract_vehicle_table(page_img: Image.Image) -> dict:
    main = crop_rel(page_img, BOX_MAIN)

    v_lines = detect_v_lines(main)
    if len(v_lines) < 3:
        W = main.size[0]
        v_lines = [int(W*0.17), int(W*0.45), int(W*0.74)]

    h_lines = detect_h_lines(main)
    ys = find_vehicle_rows(main, v_lines, h_lines)
    if ys is None:
        H = main.size[1]
        ys = [int(H*r) for r in (0.206,0.261,0.340,0.409,0.479,0.555)]

    left, mid, right = v_lines[0], v_lines[1], v_lines[2]

    def cell_block(x1,x2,y1,y2):
        c = main.crop((x1,y1,x2,y2))
        c = preprocess_for_ocr(preprocess_remove_lines(c))
        return ocr_block(c, psm=6)

    def cell_line(x1,x2,y1,y2, wl, scales=(6.0, 7.5)):
        # multi-scale: escoger la que cumpla mejor (placa especialmente)
        best = ""
        for sc in scales:
            c = main.crop((x1,y1,x2,y2))
            c = preprocess_small_line(preprocess_remove_lines(c), scale=sc)
            raw = ocr_line(c, wl, psm=7)
            if raw and len(raw) > len(best):
                best = raw
        return best

    r0l = cell_block(left, mid, ys[0], ys[1])     # Marca
    r0r = cell_block(mid, right, ys[0], ys[1])    # Chasis
    r1l = cell_block(left, mid, ys[1], ys[2])     # Linea
    r1r = cell_block(mid, right, ys[1], ys[2])    # Pasajeros
    r2l = cell_block(left, mid, ys[2], ys[3])     # Modelo
    r2r = cell_block(mid, right, ys[2], ys[3])    # Clase
    r3l = cell_line(left, mid, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", scales=(6.0,7.0,8.0))
    r3r = cell_line(mid, right, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", scales=(6.0,7.0))
    r4l = cell_block(left, mid, ys[4], ys[5])     # Color

    marca = norm_isuzu(r0l)
    chasis = fix_vin_ocr(r0r)
    linea = norm_np(r1l)
    pasajeros = norm_pas(r1r)
    modelo = norm_year(r2l)
    clase = norm_camion(r2r)
    placa = norm_placa(r3l)
    motor = norm_motor(r3r)
    color = norm_blanco(r4l)

    # validate
    if not v_marca(marca): marca = ""
    if not v_chasis(chasis): chasis = ""
    if not v_linea(linea): linea = ""
    if not v_pas(pasajeros): pasajeros = ""
    if not v_modelo(modelo): modelo = ""
    if not v_clase(clase): clase = ""
    if not v_placa(placa): placa = ""
    if not v_motor(motor): motor = ""
    if not v_color(color): color = ""

    # fallback for marca/chasis if missing (label + fixed)
    if not marca or not chasis:
        fb_m, fb_c = fallback_marca_chasis_label(main)
        if not marca and fb_m: marca = fb_m
        if not chasis and fb_c: chasis = fb_c

        if not marca:
            fm = fallback_marca_fixed(main)
            if fm: marca = fm
        if not chasis:
            fc = fallback_chasis_fixed(main)
            if fc: chasis = fc

    return {
        "marca": marca,
        "chasis": chasis,
        "linea": linea,
        "pasajeros": pasajeros,
        "modelo": modelo,
        "clase": clase,
        "placa": placa,
        "motor": motor,
        "color": color,
    }

def extract_page(doc, page_index: int) -> dict:
    page_img = render_page(doc, page_index, zoom=PAGE_ZOOM)

    cert = extract_cert(page_img)
    if not v_cert(cert): cert = ""

    veh = extract_vehicle_table(page_img)

    row = {"certificado_no": cert, **veh}
    for c in FIELDS:
        row.setdefault(c, "")
    return row

def compute_metrics(df: pd.DataFrame):
    filled = df.apply(lambda col: col.map(lambda v: str(v).strip() != ""))
    completeness = filled.mean().mean()
    by_field = filled.mean().sort_values(ascending=False)
    return completeness, by_field

def main():
    print(f"Procesando {PDF_PATH}")
    doc = fitz.open(PDF_PATH)

    rows = []
    for i in range(len(doc)):
        row = extract_page(doc, i)
        rows.append(row)

        valid_cnt = sum(1 for k in FIELDS if str(row[k]).strip() != "")
        print(
            f"  Pag {i+1:2d}: "
            f"cert={row['certificado_no']:<4} "
            f"marca={row['marca']:<6} "
            f"chasis={row['chasis']:<17} "
            f"linea={row['linea']:<3} "
            f"pas={row['pasajeros']:<2} "
            f"modelo={row['modelo']:<4} "
            f"clase={row['clase']:<6} "
            f"placa={row['placa']:<10} "
            f"motor={row['motor']:<10} "
            f"color={row['color']:<10} "
            f"(valid {valid_cnt}/10)"
        )

    df = pd.DataFrame(rows)[FIELDS]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    comp, by_field = compute_metrics(df)
    print("\nOK ->", OUT_CSV)
    print("Filas:", len(df), "| Paginas esperadas:", len(doc))
    print(f"\nCompletitud TOTAL (10 campos): {comp*100:.2f}%")
    print("\n-- Llenado por campo (post-validación) --")
    for k, v in by_field.items():
        print(f"{k:15s}: {v*100:6.2f}%")

if __name__ == "__main__":
    main()