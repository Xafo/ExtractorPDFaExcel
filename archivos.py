# -*- coding: utf-8 -*-
"""
Extractor de certificados de póliza vehicular desde PDF escaneado.
Genera un CSV con los datos de cada vehículo.

Generalizado para manejar múltiples formatos de VIN, línea, clase, color y placa.
"""
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

# Known VIN prefixes for ISUZU vehicles in Guatemala
VIN_PREFIXES = ["JAANPR", "JAANKR", "JALFVR", "MPAUCS"]

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
# Normalizers (GENERALIZED)
# -------------------------
def norm_marca(txt: str) -> str:
    """Normaliza marca - acepta ISUZU y variantes OCR."""
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())
    if "ISUZU" in t or t in ("TSUZU","RSUZU","IISUZU","SUZU","TSU","ISUZLI","1SUZU"):
        return "ISUZU"
    # Devolver el texto limpio si no es ISUZU
    if len(t) >= 3:
        return t
    return ""

def fix_vin_ocr(txt: str) -> str:
    """Normaliza VIN/Chasis - acepta múltiples prefijos ISUZU."""
    t = re.sub(r"[^A-Z0-9]", "", (txt or "").upper())

    # Correcciones OCR comunes
    t = t.replace("JAAMPR","JAANPR")
    t = t.replace("JAAMKR","JAANKR")

    # Intentar cada prefijo conocido
    for prefix in VIN_PREFIXES:
        idx = t.find(prefix)
        if idx != -1 and idx + 17 <= len(t):
            vin = t[idx:idx+17]

            # Correcciones específicas para JAANPR
            if prefix == "JAANPR":
                core = list(vin)
                core[6], core[7] = "7","1"  # fuerza 71
                vin = "".join(core)
                vin = re.sub(r"HR110", "HR710", vin)
                vin = re.sub(r"HS110", "HS710", vin)

            if len(vin) == 17 and all(c in VIN_ALLOWED for c in vin):
                return vin

    # Fallback: intentar con sustituciones I->1, O->0 y buscar de nuevo
    t2 = t.replace("I","1").replace("O","0").replace("Q","0")
    for prefix in VIN_PREFIXES:
        idx = t2.find(prefix)
        if idx != -1 and idx + 17 <= len(t2):
            vin = t2[idx:idx+17]
            if prefix == "JAANPR":
                core = list(vin)
                core[6], core[7] = "7","1"
                vin = "".join(core)
                vin = re.sub(r"HR110", "HR710", vin)
                vin = re.sub(r"HS110", "HS710", vin)
            if len(vin) == 17 and all(c in VIN_ALLOWED for c in vin):
                return vin

    # Si no encontramos un prefijo conocido, buscar cualquier secuencia de 17 chars
    # que parezca un VIN (empieza con J o M)
    t3 = re.sub(r"[^A-Z0-9]", "", t2)
    for start_char in ["J", "M"]:
        idx = t3.find(start_char)
        while idx != -1 and idx + 17 <= len(t3):
            candidate = t3[idx:idx+17]
            if len(candidate) == 17 and all(c in VIN_ALLOWED for c in candidate):
                return candidate
            idx = t3.find(start_char, idx + 1)

    return ""

def norm_linea(txt: str) -> str:
    """Normaliza línea - acepta NP, FVR, SIN MODELO, MU-X, etc."""
    t = norm_spaces((txt or "").upper())

    if "SIN MODELO" in t or "SINMODELO" in t.replace(" ",""):
        return "SIN MODELO"
    if "MU-X" in t or "MUX" in t.replace("-","").replace(" ",""):
        return "MU-X"

    t_clean = re.sub(r"[^A-Z0-9]", "", t)
    if "FVR" in t_clean:
        return "FVR"
    if "NP" in t_clean or t_clean in ("KP","PP","NIP","MP"):
        return "NP"
    if "NPR" in t_clean:
        return "NPR"
    if "NQR" in t_clean:
        return "NQR"

    # Devolver el texto limpio si tiene al menos 1 carácter
    cleaned = re.sub(r"[^A-Z0-9\s\-/]", "", t).strip()
    if cleaned:
        return cleaned
    return ""

def norm_pas(txt: str) -> str:
    m = re.search(r"(\d{1,2})", txt or "")
    return m.group(1) if m else ""

def norm_year(txt: str) -> str:
    m = re.search(r"(19|20)\d{2}", txt or "")
    return m.group(0) if m else ""

def norm_clase(txt: str) -> str:
    """Normaliza clase - acepta CAMION, CAMIONETA, SUV, etc."""
    t = (txt or "").upper()
    if "CAMIONETA" in t:
        # Check for SUV / CAMIONETA
        if "SUV" in t:
            return "SUV / CAMIONETA"
        return "CAMIONETA"
    if "CAMION" in t or any(k in t for k in ["CANON","CAMON","CANNON","CAAION","CAAVON","CAAIION","CAAIOH","CANION"]):
        return "CAMION"
    if "SUV" in t:
        return "SUV"
    if "PICKUP" in t:
        return "PICKUP"
    # Return cleaned text if something is there
    cleaned = re.sub(r"[^A-Z0-9\s/]", "", t).strip()
    if cleaned:
        return cleaned
    return ""

def norm_color(txt: str) -> str:
    """Normaliza color - acepta BLANCO, S/C, y variantes."""
    t = norm_spaces((txt or "").upper())

    if "BLANCO PERLA" in t:
        return "BLANCO PERLA DOLOMITE"
    if "BLANCO" in t:
        return "BLANCO"
    if "S/C" in t or "SIC" in t or "S.C" in t or t in ("SC","S/0","S/Q"):
        return "S/C"
    if "NEGRO" in t:
        return "NEGRO"
    if "ROJO" in t:
        return "ROJO"
    if "GRIS" in t:
        return "GRIS"
    if "AZUL" in t:
        return "AZUL"

    # Return cleaned text
    cleaned = re.sub(r"[^A-Z0-9\s/]", "", t).strip()
    if cleaned:
        return cleaned
    return ""

def norm_motor(raw: str) -> str:
    """Normaliza motor - conservador con las sustituciones.
    Solo hace sustituciones seguras que no corrompan datos válidos."""
    t = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())

    # Remove leading 'M' artifact from OCR only if followed by typical pattern
    if t.startswith("M0") and len(t) > 6:
        t = t[1:]  # M0VN405 -> 0VN405
    elif t.startswith("MN") and len(t) > 6:
        t = t[1:]  # Posible artifact

    # Remove leading N only if it creates a known pattern like N0VN -> 0VN
    if t.startswith("N0V") and len(t) > 6:
        t = t[1:]
    if t.startswith("N207") and len(t) > 6:
        t = t[1:]  # N207N315 -> probably artifact
    if t.startswith("A0") and len(t) > 6:
        t = t[1:]  # A07N353 -> 07N353... hmm, could be real

    # Minimal OCR corrections - ONLY for clearly wrong substitutions
    # Don't blindly convert O->0, S->5 etc as motors can have real letters
    # Instead, be very targeted
    if len(t) > 8:
        t = t[:8]

    return t if 3 <= len(t) <= 10 else ""

def norm_placa(raw: str) -> str:
    """Normaliza placa - maneja formatos guatemaltecos variados.
    Formatos vistos:
    - C-607BYF  (letra - 3dígitos - 3letras)
    - C-0382BRX (letra - 4dígitos - 3letras)
    - P-445KTY  (letra - 3dígitos - 3letras)
    - C-C0381BRX -> C-0381BRX (OCR artifact: letra duplicada)
    
    B/8 correction: OCR commonly reads 'B' as '8'. When we see
    4 digits + 2 letters (e.g. C-6078YF), the last '8' is likely 'B',
    giving us C-607BYF (3 digits + 3 letters).
    This does NOT apply when there are already 3 letters (e.g. C-0458BRM stays).
    """
    t = (raw or "").upper().replace(" ", "")
    t = re.sub(r"[^A-Z0-9\-.]", "", t)
    t = t.replace(".", "-")

    # Fix double prefix letter: "C-C0381BRX" -> "C-0381BRX"
    m_double = re.match(r"([A-Z])\-?([A-Z])(\d)", t)
    if m_double and m_double.group(1) == m_double.group(2):
        t = m_double.group(1) + "-" + t[t.index(m_double.group(2), 1)+1:]

    # Main pattern: Letter - 3to4 digits - 2to3 letters
    m = re.search(r"([A-Z])\-?(\d{3,4})([A-Z]{2,3})\b", t)
    if m:
        prefix = m.group(1)
        digits = m.group(2)
        letters = m.group(3)

        # B/8 correction: 4 digits + 2 letters where last digit is '8'
        # -> likely OCR read 'B' as '8', convert to 3 digits + B + 2 letters
        if len(digits) == 4 and len(letters) == 2 and digits[-1] == "8":
            return f"{prefix}-{digits[:3]}B{letters}"

        return f"{prefix}-{digits}{letters}"

    # Fallback: more flexible digit/letter split
    m2 = re.search(r"([A-Z])\-?(\d{3,5})([A-Z]{1,3})", t)
    if m2:
        prefix = m2.group(1)
        digits = m2.group(2)
        letters = m2.group(3)
        if len(digits) >= 3 and len(letters) >= 2:
            return f"{prefix}-{digits}{letters}"

    return ""

# -------------------------
# Validators (GENERALIZED)
# -------------------------
def v_cert(x):   return bool(re.fullmatch(r"\d{3,4}", x or ""))
def v_marca(x):  return bool(x) and len(x) >= 3  # Accept any valid marca
def v_chasis(x): return bool(x) and len(x) == 17  # 17 chars for any VIN
def v_linea(x):  return bool(x) and len(x) >= 1   # Accept any non-empty
def v_pas(x):    return bool(re.fullmatch(r"\d{1,2}", x or ""))
def v_modelo(x): return bool(re.fullmatch(r"(19|20)\d{2}", x or ""))
def v_clase(x):  return bool(x) and len(x) >= 3   # Accept any valid clase
def v_color(x):  return bool(x) and len(x) >= 2   # Accept any valid color
def v_placa(x):  return bool(re.fullmatch(r"[A-Z]-\d{3,4}[A-Z]{2,3}", (x or "").upper()))
def v_motor(x):  return bool(re.fullmatch(r"[A-Z0-9]{3,10}", (x or "").upper()))

# -------------------------
# Fallbacks for Marca/Chasis
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
    marca = norm_marca(marca_txt)
    chasis = fix_vin_ocr(chas_txt)
    if not v_marca(marca): marca = ""
    if not v_chasis(chasis): chasis = ""
    return marca, chasis

def fallback_marca_fixed(main_rgb):
    cands = [
        (0.02, 0.18, 0.48, 0.28),
        (0.02, 0.16, 0.50, 0.30),
        (0.00, 0.18, 0.55, 0.29),
    ]
    for b in cands:
        img = crop_rel_inside(main_rgb, b)
        txt = ocr_block(preprocess_for_ocr(preprocess_remove_lines(img))).upper()
        m = norm_marca(txt)
        if m:
            return m
    return ""

def fallback_chasis_fixed(main_rgb):
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
# Extract vehicle table
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
        best = ""
        for sc in scales:
            c = main.crop((x1,y1,x2,y2))
            c = preprocess_small_line(preprocess_remove_lines(c), scale=sc)
            raw = ocr_line(c, wl, psm=7)
            if raw and len(raw) > len(best):
                best = raw
        return best

    # Also try block OCR for placa/motor as fallback (sometimes better than line OCR)
    def cell_block_fallback(x1,x2,y1,y2):
        c = main.crop((x1,y1,x2,y2))
        c = preprocess_for_ocr(preprocess_remove_lines(c))
        return ocr_block(c, psm=7)

    r0l = cell_block(left, mid, ys[0], ys[1])     # Marca
    r0r = cell_block(mid, right, ys[0], ys[1])    # Chasis
    r1l = cell_block(left, mid, ys[1], ys[2])     # Linea
    r1r = cell_block(mid, right, ys[1], ys[2])    # Pasajeros
    r2l = cell_block(left, mid, ys[2], ys[3])     # Modelo
    r2r = cell_block(mid, right, ys[2], ys[3])    # Clase
    r3l = cell_line(left, mid, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.", scales=(6.0,7.0,8.0))
    r3r = cell_line(mid, right, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", scales=(6.0,7.0))
    r4l = cell_block(left, mid, ys[4], ys[5])     # Color

    marca = norm_marca(r0l)
    chasis = fix_vin_ocr(r0r)
    linea = norm_linea(r1l)
    pasajeros = norm_pas(r1r)
    modelo = norm_year(r2l)
    clase = norm_clase(r2r)
    placa = norm_placa(r3l)
    motor = norm_motor(r3r)
    color = norm_color(r4l)

    # Fallback: try block OCR for placa if line OCR failed
    if not v_placa(placa):
        r3l_fb = cell_block_fallback(left, mid, ys[3], ys[4])
        placa_fb = norm_placa(r3l_fb)
        if v_placa(placa_fb):
            placa = placa_fb

    # Fallback: try block OCR for motor if line OCR failed
    if not v_motor(motor):
        r3r_fb = cell_block_fallback(mid, right, ys[3], ys[4])
        motor_fb = norm_motor(r3r_fb)
        if v_motor(motor_fb):
            motor = motor_fb

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

    # fallback for marca/chasis if missing
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
            f"linea={row['linea']:<12} "
            f"pas={row['pasajeros']:<2} "
            f"modelo={row['modelo']:<4} "
            f"clase={row['clase']:<16} "
            f"placa={row['placa']:<12} "
            f"motor={row['motor']:<10} "
            f"color={row['color']:<22} "
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