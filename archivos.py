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

PAGE_ZOOM = 6.0
LANG = "spa+eng"
OCR_CFG = "--oem 1 --psm 6"

# Recorte relativo del bloque donde están los campos del vehículo (tu cuadro)
# Si tu PDF tiene pequeñas variaciones, este box suele ser estable.
BOX_MAIN = (0.37, 0.28, 0.95, 0.52)

ALL_FIELDS = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")  # sin I,O,Q

# =========================
# OCR PIPELINE
# =========================
def render_page(doc, page_index: int) -> Image.Image:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(PAGE_ZOOM, PAGE_ZOOM), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def crop_rel(img: Image.Image, box):
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))

def preprocess_remove_lines(pil_img: Image.Image) -> Image.Image:
    """Quita líneas de tabla (verticales/horizontales) para que Tesseract no se confunda."""
    gray = np.array(pil_img.convert("L"))

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    inv = 255 - th

    horiz = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)), 1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)), 1)

    vert = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)), 1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)), 1)

    lines = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.subtract(inv, lines)
    out = 255 - cleaned
    return Image.fromarray(out)

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    g = pil_img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=2)
    g = g.filter(ImageFilter.SHARPEN)
    return g

def ocr(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang=LANG, config=OCR_CFG).replace("\x0c", "")

def norm_spaces(s: str) -> str:
    return " ".join((s or "").split())

# =========================
# NORMALIZACIÓN / VALIDACIÓN
# =========================
def _filled(v):
    if v is None: return False
    if isinstance(v, float) and pd.isna(v): return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"

def normalize_cert(text: str) -> str:
    m = re.search(r"(\d{3,4})", text or "")
    return m.group(1) if m else ""

def normalize_marca(text: str) -> str:
    t = (text or "").upper()
    if "ISUZU" in t: return "ISUZU"
    t = re.sub(r"[^A-Z]", "", t)
    return t[:15]

def normalize_linea(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    if t in ("KP", "PP"):  # OCR común
        return "NP"
    return t[:4]

def normalize_pasajeros(text: str) -> str:
    m = re.search(r"(\d{1,2})", text or "")
    if not m: return ""
    v = int(m.group(1))
    return m.group(1) if 1 <= v <= 60 else ""

def normalize_modelo(text: str) -> str:
    m = re.search(r"((?:19|20)\d{2})", text or "")
    if not m: return ""
    v = int(m.group(1))
    return m.group(1) if 1990 <= v <= 2035 else ""

def normalize_clase(text: str) -> str:
    t = (text or "").upper()
    # Mapeos típicos por OCR (CANON/CAMON/etc.) => CAMION
    if "CAMION" in t:
        return "CAMION"
    if re.search(r"\bCANON\b|\bCAMON\b|\bCAMWON\b|\bAMMON\b|\bMAMON\b|\bSAMON\b|\bSON\b", t):
        return "CAMION"
    t = re.sub(r"[^A-Z]", "", t)
    return t[:15]

def normalize_placa(text: str) -> str:
    t = (text or "").upper().replace(" ", "")
    # patrones tipo C-6078YF, C.6078YF, C6078YF
    m = re.search(r"([A-Z])[\.-]?(\d{3,4}[A-Z]{2,3})", t)
    return f"{m.group(1)}-{m.group(2)}" if m else ""

def normalize_motor(text: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    # correcciones comunes
    t = t.translate(str.maketrans({
        "O":"0","Q":"0",
        "I":"1","L":"1",
        "S":"5","Z":"2",
        "B":"8",
        "U":"N",  # N a veces se lee U
        "G":"6",  # 6 a veces se lee G
    }))
    t = re.sub(r"^MOTOR", "", t)
    return t if 3 <= len(t) <= 10 else ""

def normalize_chasis(text: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    s = s.translate(str.maketrans({
        "T":"7","V":"7",
        "I":"1","L":"1",
        "O":"0","Q":"0",
        "G":"3",  # 3 suele leerse como G
    }))

    # Si aparece el prefijo observado, recorta ventana VIN
    if len(s) >= 17:
        idx = s.find("JAANPR")
        if idx != -1 and idx + 17 <= len(s):
            s = s[idx:idx+17]
        else:
            s = s[:17]

    # fuerza patrón JAANPR7...
    if s.startswith("JAANPR") and len(s) >= 7:
        s = "JAANPR7" + s[7:]

    # valida alfabeto VIN
    if len(s) == 17 and all(c in VIN_ALLOWED for c in s):
        return s
    return s[:17]

def normalize_color(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", t).strip()
    # OCR típico
    t = (t.replace("JLANCO","BLANCO")
           .replace("8LANCO","BLANCO")
           .replace("BIANCO","BLANCO")
           .replace("MANCO","BLANCO"))
    # solo primera palabra (BLANCO, ROJO, etc.)
    return t.split()[0] if t else ""

# VALIDADORES (para métrica de "validez")
def v_cert(x):
    if not _filled(x): return False
    try:
        v = int(str(x).split(".")[0])
        return 1 <= v <= 9999
    except:
        return False

def v_marca(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z]{3,15}", str(x).strip().upper()))

def v_chasis(x):
    if not _filled(x): return False
    s = re.sub(r"[^A-Z0-9]", "", str(x).upper())
    return len(s) == 17 and all(c in VIN_ALLOWED for c in s)

def v_linea(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z0-9]{1,4}", str(x).strip().upper()))

def v_pasajeros(x):
    if not _filled(x): return False
    try:
        v = int(float(x))
        return 1 <= v <= 60
    except:
        return False

def v_modelo(x):
    if not _filled(x): return False
    try:
        v = int(float(x))
        return 1990 <= v <= 2035
    except:
        return False

def v_clase(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z]{3,15}", str(x).strip().upper()))

def v_placa(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z]-\d{3,4}[A-Z]{2,3}", str(x).strip().upper()))

def v_motor(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z0-9]{3,10}", str(x).strip().upper()))

def v_color(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-ZÁÉÍÓÚÑ]{3,15}", str(x).strip().upper()))

VALIDATORS = {
    "certificado_no": v_cert,
    "marca": v_marca,
    "chasis": v_chasis,
    "linea": v_linea,
    "pasajeros": v_pasajeros,
    "modelo": v_modelo,
    "clase": v_clase,
    "placa": v_placa,
    "motor": v_motor,
    "color": v_color,
}

# =========================
# EXTRACCIÓN POR PÁGINA
# =========================
def extract_page(doc, page_index: int) -> dict:
    page_img = render_page(doc, page_index)
    main = crop_rel(page_img, BOX_MAIN)

    # quita líneas + mejora contraste
    main = preprocess_for_ocr(preprocess_remove_lines(main))

    full_text = norm_spaces(ocr(main)).upper()

    def grab(pat: str) -> str:
        m = re.search(pat, full_text)
        return m.group(1).strip() if m else ""

    # Regex tolerantes (si OCR lee CERTINCADO/M0DELO/etc., ajusta aquí)
    raw_cert = grab(r"CERT\w*ADO\s*NO\.?\s*[:\.]?\s*(\d{3,4})")
    raw_marca = grab(r"MARCA\s*:\s*([A-Z]{3,10})")
    raw_chasis = grab(r"CHASIS\s*:\s*([A-Z0-9]{10,25})")
    raw_linea = grab(r"LINEA\s*:\s*([A-Z0-9]{1,4})")
    raw_pas = grab(r"PASAJEROS\s*:\s*(\d{1,2})")
    raw_modelo = grab(r"MODELO\s*:\s*((?:19|20)\d{2})")
    raw_clase = grab(r"CLASE\s*:\s*([A-Z]{3,10})")
    raw_placa = grab(r"PLACA\s*:\s*([A-Z0-9\-\.\s]{4,15})")
    raw_motor = grab(r"MOTOR\s*:\s*([A-Z0-9\s]{3,12})")
    raw_color = grab(r"COLOR\s*:\s*([A-ZÁÉÍÓÚÑ]{3,15})")

    row = {
        "certificado_no": normalize_cert(raw_cert),
        "marca": normalize_marca(raw_marca),
        "chasis": normalize_chasis(raw_chasis),
        "linea": normalize_linea(raw_linea),
        "pasajeros": normalize_pasajeros(raw_pas),
        "modelo": normalize_modelo(raw_modelo),
        "clase": normalize_clase(raw_clase),
        "placa": normalize_placa(raw_placa),
        "motor": normalize_motor(raw_motor),
        "color": normalize_color(raw_color),
    }
    return row

# =========================
# MÉTRICAS (TODO LOS CAMPOS)
# =========================
def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    for c in ALL_FIELDS:
        if c not in df.columns:
            df[c] = None
    df = df[ALL_FIELDS]

    filled_matrix = df.apply(lambda col: col.map(_filled))
    row_fill = filled_matrix.sum(axis=1) / len(ALL_FIELDS)

    valid_matrix = pd.DataFrame({c: df[c].map(VALIDATORS[c]) for c in ALL_FIELDS})
    row_valid = valid_matrix.sum(axis=1) / len(ALL_FIELDS)

    global_fill = row_fill.mean()
    global_valid = row_valid.mean()

    fill_by_field = filled_matrix.mean().sort_values(ascending=False)
    valid_by_field = valid_matrix.mean().sort_values(ascending=False)

    return row_fill, row_valid, global_fill, global_valid, fill_by_field, valid_by_field

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

        # reporte rápido por página (fill/valid sobre 10 campos)
        tmp = pd.DataFrame([row])[ALL_FIELDS]
        row_fill, row_valid, _, _, _, _ = compute_metrics(tmp)
        fill_cnt = int(row_fill.iloc[0] * 10)
        valid_cnt = int(row_valid.iloc[0] * 10)

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
            f"motor={row['motor']:<6} "
            f"color={row['color']:<10} "
            f"(fill {fill_cnt}/10 | valid {valid_cnt}/10)"
        )

    df = pd.DataFrame(rows)[ALL_FIELDS]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # Métricas globales basadas en TODO
    row_fill, row_valid, gfill, gvalid, fill_field, valid_field = compute_metrics(df)

    print("\nOK ->", OUT_CSV)
    print("Filas:", len(df), "| Paginas esperadas:", len(doc))
    print(f"\nCompletitud TOTAL (10 campos): {gfill*100:.2f}%")
    print(f"Validez TOTAL (10 campos):     {gvalid*100:.2f}%")

    print("\n-- Completitud por campo (no vacio) --")
    for k, v in fill_field.items():
        print(f"{k:15s}: {v*100:6.2f}%")

    print("\n-- Validez por campo (pasa validacion) --")
    for k, v in valid_field.items():
        print(f"{k:15s}: {v*100:6.2f}%")

if __name__ == "__main__":
    main()