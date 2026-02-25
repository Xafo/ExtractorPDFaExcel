# -*- coding: utf-8 -*-
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# =========================
# CONFIG
# =========================
PDF_PATH = r"./pdfs/1943853451940.pdf"
OUT_CSV  = r"./certificados.csv"

# En Windows (descomenta y ajusta):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PAGE_ZOOM = 5.0
LANG = "spa+eng"

FIELDS = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]

# Bloque grande donde vive la tabla "Descripción del Vehículo" (tu cuadro amarillo)
BOX_MAIN = (0.37, 0.28, 0.95, 0.52)

# Candidatos de recorte para "Certificado No." (cabecera de la póliza)
CERT_BOXES = [
    (0.62, 0.235, 0.97, 0.32),
    (0.55, 0.22,  0.98, 0.33),
    (0.60, 0.22,  0.98, 0.35),
]

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")  # VIN sin I,O,Q

# =========================
# UTILIDADES BÁSICAS
# =========================
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

# =========================
# PREPROCESADO (QUITAR LÍNEAS)
# =========================
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

# =========================
# OCR
# =========================
def ocr_block(img: Image.Image, lang=LANG, psm=6) -> str:
    cfg = f"--oem 1 --psm {psm}"
    return pytesseract.image_to_string(img, lang=lang, config=cfg).replace("\x0c","").strip()

def ocr_line(img: Image.Image, whitelist: str, lang="eng", psm=7) -> str:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, lang=lang, config=cfg).replace("\x0c","").strip()

# =========================
# DETECCIÓN DE LÍNEAS (SEGMENTACIÓN DE LA TABLA)
# =========================
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
        # quédate con las 3 más fuertes
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
    idx = np.where(hsum > main_pil.size[0]*0.3)[0]  # width * 0.3
    return cluster_positions(idx, 2)

def find_vehicle_rows(main_pil: Image.Image, v_lines, h_lines):
    """
    Busca un bloque de 6 líneas horizontales consecutivas que contenga 5 filas:
    Marca, Línea, Modelo, Placa, Color (en la columna izquierda).
    """
    if len(v_lines) < 3 or len(h_lines) < 6:
        return None

    left, mid, _ = v_lines[0], v_lines[1], v_lines[2]
    best = None

    def score_row(ys):
        score = 0
        keys = ["MAR", "LIN", "MOD", "PLA", "COL"]
        for i in range(5):
            band = main_pil.crop((left, ys[i], mid, ys[i+1]))
            txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(band)), psm=6).upper()
            if keys[i] in txt:
                score += 3
        return score

    for j in range(len(h_lines)-5):
        ys = h_lines[j:j+6]
        sc = score_row(ys)
        if best is None or sc > best[0]:
            best = (sc, ys)

    return best[1] if best and best[0] >= 3 else None

# =========================
# NORMALIZACIÓN (CONSERVADORA)
# =========================
def norm_isuzu(txt: str) -> str:
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())
    if "ISUZU" in t or t in ("TSUZU","RSUZU","IISUZU","SUZU","TSU"):
        return "ISUZU"
    return ""

def norm_vin_from_cell(txt: str) -> str:
    """
    Extrae VIN buscando el substring JAANPR... y recorta 17.
    Evita basura como 'CHASISJAANPR...'
    """
    t = re.sub(r"[^A-Z0-9]", "", (txt or "").upper())
    t = t.replace("JAAMPR", "JAANPR")
    t = t.replace("I","1").replace("O","0").replace("Q","0")
    idx = t.find("JAANPR")
    if idx != -1 and idx + 17 <= len(t):
        vin = t[idx:idx+17]
    else:
        # fallback: toma los últimos 17 si parecen VIN
        vin = t[-17:] if len(t) >= 17 else ""
    if len(vin) == 17 and all(c in VIN_ALLOWED for c in vin):
        return vin
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
    return "CAMION" if ("CAMION" in t or any(k in t for k in ["CANON","CAMON","CANNON","CAAION","CAAION"])) else ""

def norm_blanco(txt: str) -> str:
    return "BLANCO" if "BLANCO" in (txt or "").upper() else ""

def norm_motor(raw: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    t = t.lstrip("M")  # a veces queda basura del label
    t = t.translate(str.maketrans({"O":"0","Q":"0","S":"5","I":"1","L":"1"}))
    if len(t) > 8:
        t = t[:8]
    return t if 3 <= len(t) <= 10 else ""

def norm_placa(raw: str) -> str:
    t = (raw or "").upper().replace(" ", "").replace(".", "-")
    t = re.sub(r"[^A-Z0-9\-]", "", t)

    m = re.search(r"([A-Z])\-?([A-Z0-9]{5,9})", t)
    if not m:
        return ""

    pref = m.group(1)
    body = m.group(2)

    # Corrige SOLO dígitos (primeros 3-4)
    trans = str.maketrans({"G":"6","O":"0","Q":"0","D":"0","B":"8","S":"5","Z":"2","I":"1","L":"1"})
    cand4 = re.sub(r"[^0-9]", "", body[:4].translate(trans))
    if len(cand4) == 4:
        letters = re.sub(r"[^A-Z]", "", body[4:])
        if 2 <= len(letters) <= 3:
            return f"{pref}-{cand4}{letters}"

    cand3 = re.sub(r"[^0-9]", "", body[:3].translate(trans))
    if len(cand3) == 3:
        letters = re.sub(r"[^A-Z]", "", body[3:])
        if 2 <= len(letters) <= 3:
            return f"{pref}-{cand3}{letters}"

    return ""

# =========================
# VALIDACIÓN (ANTI-BASURA)
# =========================
def v_cert(x):   return bool(re.fullmatch(r"\d{3,4}", x or ""))
def v_marca(x):  return (x or "") == "ISUZU"
def v_chasis(x): return bool(x) and len(x)==17 and x.startswith("JAANPR")
def v_linea(x):  return (x or "") == "NP"
def v_pas(x):    return bool(re.fullmatch(r"\d{1,2}", x or ""))
def v_modelo(x): return bool(re.fullmatch(r"(19|20)\d{2}", x or ""))
def v_clase(x):  return (x or "") == "CAMION"
def v_color(x):  return (x or "") == "BLANCO"
def v_placa(x):  return bool(re.fullmatch(r"[A-Z]-\d{3,4}[A-Z]{2,3}", (x or "").upper()))
def v_motor(x):  return bool(re.fullmatch(r"[A-Z0-9]{3,10}", (x or "").upper()))

# =========================
# CERTIFICADO NO. (CABECERA)
# =========================
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

# =========================
# EXTRACCIÓN POR PÁGINA (TABLA DEL VEHÍCULO)
# =========================
def extract_vehicle_table(page_img: Image.Image) -> dict:
    main = crop_rel(page_img, BOX_MAIN)

    v_lines = detect_v_lines(main)
    if len(v_lines) < 3:
        W = main.size[0]
        v_lines = [int(W*0.17), int(W*0.45), int(W*0.74)]

    h_lines = detect_h_lines(main)
    ys = find_vehicle_rows(main, v_lines, h_lines)

    if ys is None:
        # fallback (ratios de la página 1); mejor que inventar
        H = main.size[1]
        ys = [int(H*r) for r in (0.206,0.261,0.340,0.409,0.479,0.555)]

    left, mid, right = v_lines[0], v_lines[1], v_lines[2]

    def cell_block(x1,x2,y1,y2):
        c = main.crop((x1,y1,x2,y2))
        c = preprocess_for_ocr(preprocess_remove_lines(c))
        return ocr_block(c, psm=6)

    def cell_line(x1,x2,y1,y2, wl):
        c = main.crop((x1,y1,x2,y2))
        c = preprocess_small_line(preprocess_remove_lines(c), scale=6.0)
        return ocr_line(c, wl, lang="eng", psm=7)

    # 5 filas: 0 Marca/Chasis, 1 Linea/Pasaj, 2 Modelo/Clase, 3 Placa/Motor, 4 Color
    r0l = cell_block(left, mid, ys[0], ys[1])
    r0r = cell_block(mid, right, ys[0], ys[1])

    r1l = cell_block(left, mid, ys[1], ys[2])
    r1r = cell_block(mid, right, ys[1], ys[2])

    r2l = cell_block(left, mid, ys[2], ys[3])
    r2r = cell_block(mid, right, ys[2], ys[3])

    # Placa/Motor: OCR tipo línea (mucho más fiable)
    r3l = cell_line(left, mid, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
    r3r = cell_line(mid, right, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    r4l = cell_block(left, mid, ys[4], ys[5])

    marca = norm_isuzu(r0l)
    chasis = norm_vin_from_cell(r0r)
    linea = norm_np(r1l)
    pasajeros = norm_pas(r1r)
    modelo = norm_year(r2l)
    clase = norm_camion(r2r)
    placa = norm_placa(r3l)
    motor = norm_motor(r3r)
    color = norm_blanco(r4l)

    # VALIDACIÓN: si no pasa, vacío (NO BASURA)
    if not v_marca(marca): marca = ""
    if not v_chasis(chasis): chasis = ""
    if not v_linea(linea): linea = ""
    if not v_pas(pasajeros): pasajeros = ""
    if not v_modelo(modelo): modelo = ""
    if not v_clase(clase): clase = ""
    if not v_placa(placa): placa = ""
    if not v_motor(motor): motor = ""
    if not v_color(color): color = ""

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
    if not v_cert(cert):
        cert = ""

    veh = extract_vehicle_table(page_img)

    row = {"certificado_no": cert, **veh}
    # asegura columnas
    for c in FIELDS:
        row.setdefault(c, "")
    return row

# =========================
# MÉTRICAS: COMPLETITUD vs VALIDEZ REAL
# =========================
def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    for c in FIELDS:
        if c not in df.columns:
            df[c] = ""
    df = df[FIELDS]

    filled = df.apply(lambda col: col.map(lambda v: str(v).strip() != ""))
    completeness = filled.mean().mean()

    # “validez” ya está aplicada (post-validación no hay basura),
    # entonces validez = completitud post-validación.
    validity = completeness

    by_field = filled.mean().sort_values(ascending=False)
    return completeness, validity, by_field

# =========================
# MAIN
# =========================
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

    comp, val, by_field = compute_metrics(df)

    print("\nOK ->", OUT_CSV)
    print("Filas:", len(df), "| Paginas esperadas:", len(doc))
    print(f"\nCompletitud TOTAL (10 campos): {comp*100:.2f}%")
    print(f"Validez TOTAL (10 campos):     {val*100:.2f}%")

    print("\n-- Llenado por campo (post-validación) --")
    for k, v in by_field.items():
        print(f"{k:15s}: {v*100:6.2f}%")

if __name__ == "__main__":
    main()