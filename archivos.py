# -*- coding: utf-8 -*-
import os
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
LANG_MAIN = "spa+eng"

ALL_FIELDS = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]

# recorte grande donde vive el cuadro amarillo (estable en tu PDF)
BOX_MAIN = (0.37, 0.28, 0.95, 0.52)

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")  # sin I,O,Q

# =========================
# OCR / IMAGEN
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
    """Quita líneas de tabla (ayuda mucho al OCR)."""
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

def preprocess_main(pil_img: Image.Image) -> Image.Image:
    g = pil_img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=2)
    g = g.filter(ImageFilter.SHARPEN)
    return g

def norm_spaces(s: str) -> str:
    return " ".join((s or "").split())

def crop_value(main_img, left, top, right, bottom, pad=2):
    w, h = main_img.size
    l = max(0, left + pad)
    r = min(w, right - pad)
    t = max(0, top - pad)
    b = min(h, bottom + pad)
    return main_img.crop((l, t, r, b))

# =========================
# OCR ESPECIALIZADO (PLACA / MOTOR)
# =========================
def preprocess_plate(img: Image.Image) -> Image.Image:
    # Esta receta la validé: en p1 dio OCR raw '-C-607BYF' => normaliza a C-6078YF
    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=0)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=2))
    arr = np.array(g)
    arr = cv2.resize(arr, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(arr)

def preprocess_motor(img: Image.Image) -> Image.Image:
    # Validado: en p1 el recorte “OVN4D5” con PSM 7 se lee mucho mejor que con PSM 8
    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=0)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=2))
    arr = np.array(g)
    arr = cv2.resize(arr, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(arr)

def ocr_line(img: Image.Image, psm: int, whitelist: str, lang="eng") -> str:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, lang=lang, config=cfg).replace("\x0c", "").strip()

# =========================
# NORMALIZACIONES
# =========================
def normalize_marca(s: str) -> str:
    t = (s or "").upper()
    t = (t.replace("RSUZU","ISUZU")
           .replace("TSUZU","ISUZU")
           .replace("IISUZU","ISUZU")
           .replace("SUZU","ISUZU"))
    return "ISUZU" if "ISUZU" in t else re.sub(r"[^A-Z]", "", t)[:15]

def normalize_clase(s: str) -> str:
    t = (s or "").upper()
    if "CAMION" in t: return "CAMION"
    if re.search(r"\bCANON\b|\bCAMON\b|\bCANNON\b|\bCAAION\b|\bCAMWON\b|\bAMMON\b|\bMAMON\b|\bSAMON\b|\bSON\b", t):
        return "CAMION"
    return re.sub(r"[^A-Z]", "", t)[:15]

def normalize_linea(s: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (s or "").upper())
    if t.startswith("NP"): return "NP"
    if t in ("KP","PP"): return "NP"
    return t[:3]

def normalize_placa(raw: str) -> str:
    t = (raw or "").upper().replace(" ", "").replace(".", "-")
    t = re.sub(r"^[^A-Z]*", "", t)

    m = re.search(r"([A-Z])\-?([A-Z0-9]{5,8})", t)
    if not m:
        return ""

    pref = m.group(1)
    body = m.group(2)

    # intentar 4 dígitos + 2-3 letras
    if len(body) < 6:
        return ""

    trans_digits = str.maketrans({
        "B":"8","S":"5","Z":"2","O":"0","Q":"0","D":"0",
        "I":"1","L":"1","U":"7","G":"6","C":"6","A":"4"
    })
    digits = body[:4].translate(trans_digits)
    digits = re.sub(r"[^0-9]", "", digits)

    if len(digits) != 4:
        # fallback 3 dígitos
        digits = body[:3].translate(trans_digits)
        digits = re.sub(r"[^0-9]", "", digits)
        if len(digits) != 3:
            return ""
        letters = body[3:]
    else:
        letters = body[4:]

    letters = re.sub(r"[^A-Z]", "", letters)
    if not (2 <= len(letters) <= 3):
        return ""

    return f"{pref}-{digits}{letters}"

def normalize_motor(raw: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    t = t.translate(str.maketrans({"O":"0","Q":"0","S":"5","Z":"2","I":"1","L":"1","B":"8"}))

    # si OCR devuelve algo tipo OVN405I, limpiá y quedate con 5-7 chars razonables
    t = re.sub(r"[^A-Z0-9]", "", t)
    if len(t) >= 7:
        # quita basura al final típica (p.ej. una I suelta)
        t = t[:7]
    return t if 3 <= len(t) <= 10 else ""

def normalize_chasis(raw: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    s = s.translate(str.maketrans({"T":"7","V":"7","I":"1","L":"1","O":"0","Q":"0","G":"3"}))
    if len(s) >= 17:
        idx = s.find("JAANPR")
        if idx != -1 and idx + 17 <= len(s):
            s = s[idx:idx+17]
        else:
            s = s[:17]
    if s.startswith("JAANPR") and len(s) >= 7:
        s = "JAANPR7" + s[7:]
    return s[:17]

def normalize_color(s: str) -> str:
    t = re.sub(r"[^A-ZÁÉÍÓÚÑ ]", "", (s or "").upper()).strip()
    t = (t.replace("JLANCO","BLANCO").replace("8LANCO","BLANCO").replace("BIANCO","BLANCO"))
    return t.split()[0] if t else ""

# =========================
# MÉTRICAS 10/10
# =========================
def _filled(v):
    if v is None: return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"

def v_placa(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z]-\d{3,4}[A-Z]{2,3}", str(x).strip().upper()))

def v_motor(x):
    if not _filled(x): return False
    return bool(re.fullmatch(r"[A-Z0-9]{3,10}", str(x).strip().upper()))

def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    for c in ALL_FIELDS:
        if c not in df.columns:
            df[c] = ""
    df = df[ALL_FIELDS]

    filled_matrix = df.apply(lambda col: col.map(_filled))
    row_fill = filled_matrix.sum(axis=1) / len(ALL_FIELDS)
    global_fill = row_fill.mean()

    # “Validez” estricta solo para lo crítico (placa/motor); lo demás lo dejamos como filled
    valid_matrix = filled_matrix.copy()
    valid_matrix["placa"] = df["placa"].map(v_placa)
    valid_matrix["motor"] = df["motor"].map(v_motor)

    row_valid = valid_matrix.sum(axis=1) / len(ALL_FIELDS)
    global_valid = row_valid.mean()

    fill_by_field = filled_matrix.mean().sort_values(ascending=False)
    valid_by_field = valid_matrix.mean().sort_values(ascending=False)

    return row_fill, row_valid, global_fill, global_valid, fill_by_field, valid_by_field

# =========================
# Detección de etiquetas (dinámica)
# =========================
def get_label_bboxes(main_clean: Image.Image):
    df = pytesseract.image_to_data(
        main_clean, lang=LANG_MAIN, config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DATAFRAME
    )
    df = df.dropna(subset=["text"]).copy()
    df["u"] = df["text"].str.upper().str.replace("|", "", regex=False)

    def pick(key):
        cand = df[df["u"].str.contains(key)]
        if cand.empty:
            return None
        cand = cand.sort_values(["conf","width"], ascending=[False,False]).iloc[0]
        return int(cand.left), int(cand.top), int(cand.left+cand.width), int(cand.top+cand.height)

    keys = ["MARCA","CHASIS","LINEA","PASAJ","MODELO","CLASE","PLACA","MOTOR","COLOR"]
    return {k: pick(k) for k in keys}

# =========================
# EXTRACCIÓN POR PÁGINA
# =========================
def extract_page(doc, page_index: int) -> dict:
    page_img = render_page(doc, page_index)
    main = crop_rel(page_img, BOX_MAIN)

    main_clean = preprocess_main(preprocess_remove_lines(main))
    bboxes = get_label_bboxes(main_clean)

    full = norm_spaces(
        pytesseract.image_to_string(main_clean, lang=LANG_MAIN, config="--oem 1 --psm 6")
    ).upper()

    def grab(pat):
        m = re.search(pat, full)
        return m.group(1).strip() if m else ""

    row = {
        "certificado_no": grab(r"CERT\w*ADO\s*NO\.?\s*[:\.]?\s*(\d{3,4})"),
        "marca": normalize_marca(grab(r"MARCA\s*:\s*([A-Z]{3,10})")),
        "chasis": normalize_chasis(grab(r"CHASIS\s*:\s*([A-Z0-9]{10,25})")),
        "pasajeros": grab(r"PASAJEROS\s*:\s*(\d{1,2})"),
        "modelo": grab(r"MODELO\s*:\s*((?:19|20)\d{2})"),
        "clase": normalize_clase(grab(r"CLASE\s*:\s*([A-Z]{3,10})")),
        "color": normalize_color(grab(r"COLOR\s*:\s*([A-ZÁÉÍÓÚÑ]{3,15})")),
        "linea": "",
        "placa": "",
        "motor": "",
    }

    w, h = main.size

    def bbox_crop(lbl_key, right_bound_key=None, right_fallback=None):
        bb = bboxes.get(lbl_key)
        if bb is None:
            return None
        l,t,r,b = bb
        rb = bboxes.get(right_bound_key) if right_bound_key else None
        right = rb[0] if rb else (right_fallback if right_fallback is not None else w)
        return crop_value(main, r, t, right, b, pad=2)

    # LINEA: recorte a la derecha de "Linea:" hasta antes de "Pasajeros:" (si existe)
    linea_img = bbox_crop("LINEA", "PASAJ", right_fallback=int(w*0.55))
    if linea_img:
        txt = ocr_line(preprocess_motor(preprocess_remove_lines(linea_img)),
                       psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", lang="eng")
        row["linea"] = normalize_linea(txt)

    # PLACA: recorte a la derecha de "Placa:" hasta antes de "Motor:"
    placa_img = bbox_crop("PLACA", "MOTOR", right_fallback=int(w*0.55))
    if placa_img:
        raw = ocr_line(preprocess_plate(preprocess_remove_lines(placa_img)),
                       psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", lang="eng")
        row["placa"] = normalize_placa(raw)

    # MOTOR: recorte a la derecha de "Motor:" hasta el final de la columna
    motor_img = bbox_crop("MOTOR", None, right_fallback=w)
    if motor_img:
        # probé: PSM 7 da lecturas más “cortas y correctas” que PSM 8 en este campo
        raw7 = ocr_line(preprocess_motor(preprocess_remove_lines(motor_img)),
                        psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", lang="eng")
        raw8 = ocr_line(preprocess_motor(preprocess_remove_lines(motor_img)),
                        psm=8, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", lang="eng")
        # elegir el raw más “razonable” (preferir 5-7 chars)
        cand = [raw7, raw8]
        cand = sorted(cand, key=lambda s: (0 if 5 <= len(re.sub(r"[^A-Z0-9]","",s)) <= 7 else 1, len(s)))
        row["motor"] = normalize_motor(cand[0])

    return row

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
            f"motor={row['motor']:<10} "
            f"color={row['color']:<10} "
            f"(fill {fill_cnt}/10 | valid {valid_cnt}/10)"
        )

    df = pd.DataFrame(rows)[ALL_FIELDS]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

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