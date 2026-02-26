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

PDF_PATH = r"./pdfs/"
OUT_CSV  = r"./certificados.csv"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PAGE_ZOOM = 5.0
LANG = "spa+eng"
FIELDS = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]

BOX_MAIN_CANDIDATES = [
    (0.37, 0.28, 0.95, 0.52),
]
CERT_BOXES = [
    (0.62, 0.235, 0.97, 0.32),
    (0.55, 0.22,  0.98, 0.33),
    (0.60, 0.22,  0.98, 0.35),
    (0.50, 0.21,  0.98, 0.36),
]

CERT_BOXES_FALLBACK = [
    (0.36, 0.12, 0.98, 0.36),
    (0.28, 0.10, 0.98, 0.40),
]

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

# Known VIN prefixes for ISUZU vehicles in Guatemala
VIN_PREFIXES = ["JAANPR", "JAANKR", "JALFVR", "MPAUCS"]

LABEL_NOISE = {
    "MARCA", "CHASIS", "LINEA", "PASAJEROS", "MODELO", "CLASE", "PLACA", "MOTOR", "COLOR",
    "FORMA", "PAGO", "EFECTIVO", "QUETZAL", "QUETZALES", "DESCRIPCION", "VEHICULO", "DATOS",
}

KNOWN_MARCAS = {
    "ISUZU", "INTERNATIONAL", "SUZUKI", "HINO", "GREAT", "GREATDANE", "WALL", "TOYOTA", "NISSAN", "HONDA",
    "MITSUBISHI", "MAZDA", "HYUNDAI", "KIA", "CHEVROLET", "FORD", "VOLKSWAGEN", "MERCEDESBENZ",
    "UTILITY",
}

KNOWN_CLASES = {
    "CAMION", "CAMIONETA", "SUV", "SUV / CAMIONETA", "PICKUP", "AUTOMOVIL", "PANEL", "FURGON",
    "MICROBUS", "BUS", "MOTO", "OTROS",
}

KNOWN_COLORS = {
    "BLANCO", "NEGRO", "ROJO", "GRIS", "AZUL", "S/C", "PLATEADO", "PLATEADO METALICO",
    "BLANCO PERLA DOLOMITE", "BLANCO Y CALCOMANIA MULTICOLOR",
}

MARCA_ALIAS = {
    "TSUZU": "ISUZU",
    "IISUZU": "ISUZU",
    "1SUZU": "ISUZU",
    "SUZU": "ISUZU",
    "ISUZLI": "ISUZU",
    "LSUZU": "ISUZU",
    "GREATDANE": "GREATDANE",
}

FIELD_WEIGHTS = {
    "certificado_no": 1.8,
    "marca": 1.0,
    "chasis": 1.4,
    "linea": 0.8,
    "pasajeros": 0.8,
    "modelo": 1.1,
    "clase": 0.9,
    "placa": 1.0,
    "motor": 0.9,
    "color": 0.7,
}

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
    """Normaliza marca con reglas estrictas para evitar ruido OCR."""
    raw = (txt or "").upper()
    raw = re.sub(r"\s+", " ", raw).strip()
    for stopper in ("CHASIS", "LINEA", "PASAJEROS", "MODELO", "CLASE", "PLACA", "MOTOR", "COLOR"):
        idx = raw.find(stopper)
        if idx > 0:
            raw = raw[:idx]
            break

    t = re.sub(r"[^A-Z]", "", raw)
    if not t:
        return ""

    for bad, good in MARCA_ALIAS.items():
        t = t.replace(bad, good)

    candidates = sorted(KNOWN_MARCAS, key=len, reverse=True)
    for brand in candidates:
        if brand in t:
            return brand

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

    cleaned = re.sub(r"[^A-Z0-9\-/]", "", t)
    cleaned = cleaned.replace("LINEA", "")
    cleaned = cleaned.strip("-/ ")
    if 1 <= len(cleaned) <= 12:
        if cleaned not in LABEL_NOISE:
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
    cleaned = re.sub(r"[^A-Z0-9\s/]", "", t).strip()
    if cleaned in KNOWN_CLASES:
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

    for c in ["PLATEADO METALICO", "PLATEADO", "BLANCO", "NEGRO", "ROJO", "GRIS", "AZUL"]:
        if c in t:
            return c

    cleaned = re.sub(r"[^A-Z0-9\s/]", "", t).strip()
    if cleaned in KNOWN_COLORS:
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
def _alpha_tokens(txt: str):
    return [t for t in re.split(r"\s+", re.sub(r"[^A-Z\s]", " ", (txt or "").upper())) if t]


def has_label_noise(txt: str) -> bool:
    tokens = _alpha_tokens(txt)
    if not tokens:
        return False
    return any(t in LABEL_NOISE for t in tokens)


def has_label_noise_substring(txt: str) -> bool:
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())
    if not t:
        return False
    for bad in LABEL_NOISE:
        if bad in t:
            return True
    return False


def v_cert(x):
    return bool(re.fullmatch(r"\d{1,4}", x or ""))


def v_marca(x):
    t = re.sub(r"[^A-Z]", "", (x or "").upper())
    if not t or t in LABEL_NOISE or t.startswith("PLAC"):
        return False
    return t in KNOWN_MARCAS


def v_chasis(x):
    return bool(x) and len(x) == 17 and not has_label_noise(x)


def v_linea(x):
    t = norm_spaces((x or "").upper())
    if not t or has_label_noise(t) or has_label_noise_substring(t):
        return False
    if len(t) > 20:
        return False
    return bool(re.fullmatch(r"[A-Z0-9\s\-/]{1,20}", t))


def v_pas(x):
    if not re.fullmatch(r"\d{1,2}", x or ""):
        return False
    return 1 <= int(x) <= 60


def v_modelo(x):
    if not re.fullmatch(r"(19|20)\d{2}", x or ""):
        return False
    y = int(x)
    return 1980 <= y <= 2035


def v_clase(x):
    t = norm_spaces((x or "").upper())
    if not t or has_label_noise(t):
        return False
    return t in KNOWN_CLASES


def v_color(x):
    t = norm_spaces((x or "").upper())
    if not t or has_label_noise(t):
        return False
    return t in KNOWN_COLORS


def v_placa(x):
    return bool(re.fullmatch(r"[A-Z]-\d{3,4}[A-Z]{2,3}", (x or "").upper()))


def v_motor(x):
    t = (x or "").upper()
    if has_label_noise(t):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{3,10}", t))


def validate_vehicle_row(row: dict) -> dict:
    out = dict(row)
    checks = {
        "marca": v_marca,
        "chasis": v_chasis,
        "linea": v_linea,
        "pasajeros": v_pas,
        "modelo": v_modelo,
        "clase": v_clase,
        "placa": v_placa,
        "motor": v_motor,
        "color": v_color,
    }
    for k, fn in checks.items():
        if not fn(out.get(k, "")):
            out[k] = ""
    return out


def score_vehicle_row(row: dict) -> float:
    score = 0.0
    checks = {
        "marca": v_marca,
        "chasis": v_chasis,
        "linea": v_linea,
        "pasajeros": v_pas,
        "modelo": v_modelo,
        "clase": v_clase,
        "placa": v_placa,
        "motor": v_motor,
        "color": v_color,
    }
    for k, fn in checks.items():
        if fn(row.get(k, "")):
            score += FIELD_WEIGHTS.get(k, 1.0)
    return score


def merge_vehicle_rows(primary: dict, secondary: dict) -> dict:
    merged = {}
    for k in ["marca", "chasis", "linea", "pasajeros", "modelo", "clase", "placa", "motor", "color"]:
        p = (primary or {}).get(k, "")
        s = (secondary or {}).get(k, "")
        if p and not s:
            merged[k] = p
        elif s and not p:
            merged[k] = s
        elif p and s:
            merged[k] = p if len(p) >= len(s) else s
        else:
            merged[k] = ""
    return merged

# -------------------------
# Fallbacks for Marca/Chasis
# -------------------------
def find_label_bbox(df, key):
    cand = df[df["u"].str.contains(key, regex=True)]
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
    x2 = min(w, max(0, x2))
    if x2 <= x1 + 2:
        return None
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


def find_vehicle_rows_by_labels(main_pil: Image.Image, v_lines):
    clean = preprocess_for_ocr(preprocess_remove_lines(main_pil))
    df = ocr_data(clean)
    if df.empty or len(v_lines) < 2:
        return None

    left_limit = v_lines[1]
    ordered_patterns = [r"MAR", r"LIN", r"MOD", r"PLA", r"COL"]
    y_centers = []
    prev_y = -1

    for pat in ordered_patterns:
        cand = df[(df["left"] < left_limit) & (df["u"].str.contains(pat, regex=True))]
        if prev_y >= 0:
            cand = cand[cand["top"] >= prev_y - 15]
        if cand.empty:
            return None
        cand = cand.sort_values(["conf", "width"], ascending=[False, False]).head(4)
        best = cand.sort_values(["top", "left"], ascending=[True, True]).iloc[0]
        y = int(best.top + (best.height * 0.5))
        y_centers.append(y)
        prev_y = y

    if len(y_centers) != 5:
        return None

    H = main_pil.size[1]
    ys = [0] * 6
    first_gap = max(10, int((y_centers[1] - y_centers[0]) * 0.55))
    ys[0] = max(0, y_centers[0] - first_gap)
    for i in range(1, 5):
        ys[i] = int((y_centers[i-1] + y_centers[i]) / 2)
    last_gap = max(12, int((y_centers[4] - y_centers[3]) * 0.75))
    ys[5] = min(H, y_centers[4] + last_gap)

    ys = sorted(ys)
    if len(set(ys)) < 6:
        return None
    return ys


def extract_by_label_bboxes(main_pil: Image.Image, v_lines) -> dict:
    clean = preprocess_for_ocr(preprocess_remove_lines(main_pil))
    df = ocr_data(clean)
    if df.empty:
        return {k: "" for k in ["marca", "chasis", "linea", "pasajeros", "modelo", "clase", "placa", "motor", "color"]}

    w = main_pil.size[0]
    mid = v_lines[1] if len(v_lines) >= 2 else int(w * 0.45)
    right = v_lines[2] if len(v_lines) >= 3 else int(w * 0.78)

    boxes = {
        "marca": find_label_bbox(df, r"MAR"),
        "chasis": find_label_bbox(df, r"CHAS"),
        "linea": find_label_bbox(df, r"LIN"),
        "pasajeros": find_label_bbox(df, r"PASA"),
        "modelo": find_label_bbox(df, r"MOD"),
        "clase": find_label_bbox(df, r"CLAS"),
        "placa": find_label_bbox(df, r"PLA"),
        "motor": find_label_bbox(df, r"MOT"),
        "color": find_label_bbox(df, r"COL"),
    }

    def text_after(k, right_limit):
        b = boxes.get(k)
        img = crop_right_of_label(main_pil, b, right_limit=right_limit)
        if img is None:
            return ""
        return ocr_block(preprocess_for_ocr(preprocess_remove_lines(img)), psm=7)

    raw = {
        "marca": text_after("marca", mid),
        "chasis": text_after("chasis", right),
        "linea": text_after("linea", mid),
        "pasajeros": text_after("pasajeros", right),
        "modelo": text_after("modelo", mid),
        "clase": text_after("clase", right),
        "placa": text_after("placa", mid),
        "motor": text_after("motor", right),
        "color": text_after("color", mid),
    }

    row = {
        "marca": norm_marca(raw["marca"]),
        "chasis": fix_vin_ocr(raw["chasis"]),
        "linea": norm_linea(raw["linea"]),
        "pasajeros": norm_pas(raw["pasajeros"]),
        "modelo": norm_year(raw["modelo"]),
        "clase": norm_clase(raw["clase"]),
        "placa": norm_placa(raw["placa"]),
        "motor": norm_motor(raw["motor"]),
        "color": norm_color(raw["color"]),
    }
    return validate_vehicle_row(row)

# -------------------------
# Cert extract
# -------------------------
def extract_cert(page_img: Image.Image) -> str:
    def pick_cert(txt: str) -> str:
        t = (txt or "").upper()
        pats = [
            r"CERT\w*\s*(?:NO|NRO|NUMERO|N[O0])\.?\s*[:\.-]?\s*(\d{1,4})",
            r"CERT\w*[^\d]{0,25}(\d{1,4})",
        ]
        for pat in pats:
            m = re.search(pat, t)
            if m:
                return m.group(1)
        return ""

    w, h = page_img.size
    for box in CERT_BOXES + CERT_BOXES_FALLBACK:
        crop = page_img.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
        for clean in (preprocess_for_ocr(crop), preprocess_for_ocr(preprocess_remove_lines(crop))):
            for psm in (6, 11):
                txt = ocr_block(clean, psm=psm).upper()
                cert = pick_cert(txt)
                if cert:
                    return cert

    top = page_img.crop((0, 0, w, int(h * 0.42)))
    for clean_top in (preprocess_for_ocr(top), preprocess_for_ocr(preprocess_remove_lines(top))):
        for psm in (6, 11):
            cert = pick_cert(ocr_block(clean_top, psm=psm))
            if cert:
                return cert

    return ""

# -------------------------
# Extract vehicle table
# -------------------------
def extract_vehicle_table(page_img: Image.Image) -> dict:
    def extract_by_grid(main):
        v_lines = detect_v_lines(main)
        if len(v_lines) < 3:
            W = main.size[0]
            v_lines = [int(W*0.17), int(W*0.45), int(W*0.74)]

        h_lines = detect_h_lines(main)
        ys = find_vehicle_rows(main, v_lines, h_lines)
        if ys is None:
            ys = find_vehicle_rows_by_labels(main, v_lines)
        if ys is None:
            H = main.size[1]
            ys = [int(H*r) for r in (0.206,0.261,0.340,0.409,0.479,0.555)]

        left, mid, right = v_lines[0], v_lines[1], v_lines[2]

        def cell_block(x1,x2,y1,y2, psm=6):
            c = main.crop((x1,y1,x2,y2))
            c = preprocess_for_ocr(preprocess_remove_lines(c))
            return ocr_block(c, psm=psm)

        def cell_line(x1,x2,y1,y2, wl, scales=(6.0, 7.5)):
            best = ""
            for sc in scales:
                c = main.crop((x1,y1,x2,y2))
                c = preprocess_small_line(preprocess_remove_lines(c), scale=sc)
                raw = ocr_line(c, wl, psm=7)
                if raw and len(raw) > len(best):
                    best = raw
            return best

        raw = {
            "marca": cell_block(left, mid, ys[0], ys[1]),
            "chasis": cell_block(mid, right, ys[0], ys[1]),
            "linea": cell_block(left, mid, ys[1], ys[2]),
            "pasajeros": cell_block(mid, right, ys[1], ys[2]),
            "modelo": cell_block(left, mid, ys[2], ys[3]),
            "clase": cell_block(mid, right, ys[2], ys[3]),
            "placa": cell_line(left, mid, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.", scales=(6.0,7.0,8.0)),
            "motor": cell_line(mid, right, ys[3], ys[4], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", scales=(6.0,7.0)),
            "color": cell_block(left, mid, ys[4], ys[5]),
        }

        placa_fb = cell_block(left, mid, ys[3], ys[4], psm=7)
        motor_fb = cell_block(mid, right, ys[3], ys[4], psm=7)

        row = {
            "marca": norm_marca(raw["marca"]),
            "chasis": fix_vin_ocr(raw["chasis"]),
            "linea": norm_linea(raw["linea"]),
            "pasajeros": norm_pas(raw["pasajeros"]),
            "modelo": norm_year(raw["modelo"]),
            "clase": norm_clase(raw["clase"]),
            "placa": norm_placa(raw["placa"]),
            "motor": norm_motor(raw["motor"]),
            "color": norm_color(raw["color"]),
        }
        if not v_placa(row["placa"]):
            row["placa"] = norm_placa(placa_fb)
        if not v_motor(row["motor"]):
            row["motor"] = norm_motor(motor_fb)

        return validate_vehicle_row(row), v_lines

    main = crop_rel(page_img, BOX_MAIN_CANDIDATES[0])
    grid_row, _ = extract_by_grid(main)

    if not grid_row.get("marca") or not grid_row.get("chasis"):
        fb_m, fb_c = fallback_marca_chasis_label(main)
        if not grid_row.get("marca") and fb_m:
            grid_row["marca"] = fb_m
        if not grid_row.get("chasis") and fb_c:
            grid_row["chasis"] = fb_c
        if not grid_row.get("marca"):
            grid_row["marca"] = fallback_marca_fixed(main)
        if not grid_row.get("chasis"):
            grid_row["chasis"] = fallback_chasis_fixed(main)

    return validate_vehicle_row(grid_row)


def extract_vehicle_top_text(page_img: Image.Image) -> dict:
    w, h = page_img.size
    crops = [
        page_img.crop((int(0.18*w), int(0.18*h), int(0.98*w), int(0.50*h))),
        page_img.crop((int(0.20*w), int(0.15*h), int(0.98*w), int(0.52*h))),
    ]

    def parse_text(raw: str):
        txt = (raw or "").upper()
        txt = txt.replace("\n", " ")
        txt = re.sub(r"\s+", " ", txt)

        def rgx(pat):
            m = re.search(pat, txt)
            return m.group(1).strip() if m else ""

        row = {
            "marca": norm_marca(rgx(r"MARCA\s*[:\-]?\s*([A-Z0-9\-\s]{2,30})")),
            "chasis": fix_vin_ocr(rgx(r"CHAS(?:IS)?\s*[:\-]?\s*([A-Z0-9\-\s]{10,25})")),
            "linea": norm_linea(rgx(r"L[IÍ]NEA\s*[:\-]?\s*([A-Z0-9\-\s]{1,20})")),
            "pasajeros": norm_pas(rgx(r"PASAJEROS\s*[:\-]?\s*([0-9]{1,2})")),
            "modelo": norm_year(rgx(r"MODELO\s*[:\-]?\s*((?:19|20)?\d{2,4})")),
            "clase": norm_clase(rgx(r"CLASE\s*[:\-]?\s*([A-Z0-9/\-\s]{3,25})")),
            "placa": norm_placa(rgx(r"PLACA\s*[:\-]?\s*([A-Z0-9\-\.\s]{4,16})")),
            "motor": norm_motor(rgx(r"MOTOR\s*[:\-]?\s*([A-Z0-9\-\s]{3,14})")),
            "color": norm_color(rgx(r"COLOR\s*[:\-]?\s*([A-Z0-9/\-\s]{2,40})")),
        }
        return validate_vehicle_row(row)

    best = {k: "" for k in ["marca", "chasis", "linea", "pasajeros", "modelo", "clase", "placa", "motor", "color"]}
    best_sc = -1.0
    for crop in crops:
        base = preprocess_for_ocr(crop)
        for psm in (6, 11):
            txt = ocr_block(base, psm=psm)
            row = parse_text(txt)
            sc = score_vehicle_row(row)
            if sc > best_sc:
                best_sc = sc
                best = row
    return best

def extract_page(doc, page_index: int) -> dict:
    page_img = render_page(doc, page_index, zoom=PAGE_ZOOM)

    cert = extract_cert(page_img)
    if not v_cert(cert): cert = ""

    veh_table = extract_vehicle_table(page_img)
    veh_text = extract_vehicle_top_text(page_img)
    veh = validate_vehicle_row(merge_vehicle_rows(veh_table, veh_text))

    row = {"certificado_no": cert, **veh}
    for c in FIELDS:
        row.setdefault(c, "")
    return row

def compute_metrics(df: pd.DataFrame):
    filled = df.apply(lambda col: col.map(lambda v: str(v).strip() != ""))
    completeness = filled.mean().mean()
    by_field = filled.mean().sort_values(ascending=False)
    return completeness, by_field


def _mode_non_empty(series) -> str:
    if series is None:
        return ""
    vals = [str(v).strip() for v in series.tolist() if str(v).strip()]
    if not vals:
        return ""
    return str(pd.Series(vals).value_counts().idxmax())


def aggressive_post_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pas_series = df["pasajeros"] if "pasajeros" in df.columns else pd.Series(dtype=str)
    mod_series = df["modelo"] if "modelo" in df.columns else pd.Series(dtype=str)
    lin_series = df["linea"] if "linea" in df.columns else pd.Series(dtype=str)

    mode_pas = _mode_non_empty(pas_series) or "3"
    mode_modelo = _mode_non_empty(mod_series)
    mode_linea = _mode_non_empty(lin_series)

    for i in df.index:
        marca = str(df.at[i, "marca"]).strip().upper()
        chasis = str(df.at[i, "chasis"]).strip().upper()
        clase = str(df.at[i, "clase"]).strip().upper()
        linea = str(df.at[i, "linea"]).strip().upper()
        pasajeros = str(df.at[i, "pasajeros"]).strip()
        modelo = str(df.at[i, "modelo"]).strip()
        placa = str(df.at[i, "placa"]).strip().upper()
        motor = str(df.at[i, "motor"]).strip().upper()

        if not linea:
            if chasis.startswith("JALFVR"):
                linea = "FVR"
            elif chasis.startswith("JAANPR") or chasis.startswith("JAANKR"):
                linea = "NP"
            elif mode_linea:
                linea = mode_linea

        if not clase and (marca == "ISUZU" or chasis.startswith("JAA")):
            clase = "CAMION"

        if not pasajeros and (clase == "CAMION" or marca == "ISUZU"):
            pasajeros = mode_pas

        if not modelo and (marca == "ISUZU" or chasis.startswith("JAA")) and mode_modelo:
            modelo = mode_modelo

        if not placa and motor:
            m = re.fullmatch(r"([A-Z])(\d{3,4})([A-Z]{2,3})[A-Z0-9]?", motor)
            if m:
                placa_guess = norm_placa(f"{m.group(1)}-{m.group(2)}{m.group(3)}")
                if v_placa(placa_guess):
                    placa = placa_guess

        df.at[i, "linea"] = linea if v_linea(linea) else ""
        df.at[i, "clase"] = clase if v_clase(clase) else ""
        df.at[i, "pasajeros"] = pasajeros if v_pas(pasajeros) else ""
        df.at[i, "modelo"] = modelo if v_modelo(modelo) else ""
        df.at[i, "placa"] = placa if v_placa(placa) else ""

    return df

def main():
    import os
    import glob

    # Carpeta donde están los PDFs (ajustá si tu ruta es distinta)
    pdf_dir = "./pdfs"
    out_csv = OUT_CSV  # usa tu constante existente, ej: "./certificados.csv"

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No encontré PDFs en: {pdf_dir}")
        return

    all_rows = []
    total_pages_expected = 0

    for pdf_path in pdf_paths:
        print(f"\nProcesando {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages_expected += len(doc)

        for i in range(len(doc)):
            row = extract_page(doc, i)

            # (Opcional) agrega el nombre del archivo para rastrear de dónde salió cada fila
            row["archivo_pdf"] = os.path.basename(pdf_path)

            all_rows.append(row)

            valid_cnt = sum(1 for k in FIELDS if str(row.get(k, "")).strip() != "")
            print(
                f"  Pag {i+1:2d}: "
                f"cert={row.get('certificado_no',''):<4} "
                f"marca={row.get('marca',''):<10} "
                f"chasis={row.get('chasis',''):<17} "
                f"linea={row.get('linea',''):<12} "
                f"pas={row.get('pasajeros',''):<2} "
                f"modelo={row.get('modelo',''):<4} "
                f"clase={row.get('clase',''):<16} "
                f"placa={row.get('placa',''):<12} "
                f"motor={row.get('motor',''):<10} "
                f"color={row.get('color',''):<22} "
                f"(valid {valid_cnt}/10)"
            )

    # Si agregaste archivo_pdf, lo incluimos en el CSV al inicio
    final_fields = (["archivo_pdf"] + FIELDS) if all_rows and "archivo_pdf" in all_rows[0] else FIELDS

    df = pd.DataFrame(all_rows)
    # asegurar columnas y orden
    for col in final_fields:
        if col not in df.columns:
            df[col] = ""
    df = df[final_fields]

    # Relleno agresivo orientado al formato de póliza actual
    if all(c in df.columns for c in FIELDS):
        df[FIELDS] = aggressive_post_fill(df[FIELDS])

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Métricas solo sobre los 10 campos originales (no cuenta archivo_pdf)
    df_metrics = df[FIELDS].copy()
    comp, by_field = compute_metrics(df_metrics)

    print("\nOK ->", out_csv)
    print("Filas:", len(df), "| Paginas esperadas:", total_pages_expected)
    print(f"\nCompletitud TOTAL (10 campos): {comp*100:.2f}%")
    print("\n-- Llenado por campo (post-validación) --")
    for k, v in by_field.items():
        print(f"{k:15s}: {v*100:6.2f}%")
        
if __name__ == "__main__":
    main()
