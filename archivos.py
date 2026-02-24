import os
import re
import glob
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ========= CONFIG =========
INPUT_DIR = r"./pdfs"
OUTPUT_CSV = r"./certificados.csv"
LANG = "spa"

# Si tesseract no está en PATH, descomentá y ajustá:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Reintentos: si no encuentra ancla o faltan campos clave, sube zoom
ZOOMS = [2.6, 3.0, 3.4]

# OCR
CFG_DATA = "--oem 1 --psm 11"  # para localizar palabras (sparse)
CFG_TEXT = "--oem 1 --psm 6 -c preserve_interword_spaces=1"  # para extraer texto del cuadro
TARGET_W = 2400
# =========================


def preprocess(img: Image.Image, target_w=TARGET_W) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    if w > target_w:
        s = target_w / w
        img = img.resize((int(w * s), int(h * s)))
    img = ImageOps.autocontrast(img)
    return img


def render_page(pdf_path: str, page_index: int, zoom: float) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def norm(s: str) -> str:
    s = (s or "").upper()
    s = s.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_code(s: str) -> str:
    s = norm(s).replace(" ", "")
    # correcciones típicas OCR (ojo: no siempre aplica, pero ayuda)
    trans = str.maketrans({"O": "0", "I": "1", "L": "1"})
    return s.translate(trans)


def find_anchor_box(page_img: Image.Image):
    """
    Busca el título del cuadro: 'DATOS GENERALES DE LA POLIZA'
    Devuelve un rect (x1,y1,x2,y2) en pixeles alrededor del cuadro completo.
    """
    img = preprocess(page_img, target_w=1800)  # rápido para localizar
    df = pytesseract.image_to_data(img, lang=LANG, config=CFG_DATA, output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    df["t"] = df["text"].astype(str).map(norm)

    # Palabras ancla (tolerante a OCR)
    key1 = df[df["t"].str.contains(r"\bDATOS\b", regex=True, na=False)]
    key2 = df[df["t"].str.contains(r"\bGENERA", regex=True, na=False)]  # GENERALES (trunc)
    key3 = df[df["t"].str.contains(r"\bPOLI", regex=True, na=False)]    # POLIZA (trunc)

    if key1.empty or key2.empty or key3.empty:
        return None, img.size, page_img.size  # no encontró ancla

    # Tomamos la línea/bloque superior más probable: el "DATOS GENERALES..."
    # Estrategia: usar el candidato de "DATOS" más alto (menor top)
    top_row = key1.sort_values("top").iloc[0]
    top_y = int(top_row["top"])

    # Como la imagen fue escalada, necesitamos factor para volver al tamaño original del page_img
    w_scaled, h_scaled = img.size
    w_orig, h_orig = page_img.size
    sx = w_orig / w_scaled
    sy = h_orig / h_scaled

    # Definimos un recorte "grande" alrededor del cuadro tomando como referencia top_y del título.
    # En tu layout, el cuadro amarillo empieza un poco arriba del título y termina bastante abajo.
    y1 = max(0, int((top_y - 30) * sy))
    y2 = min(h_orig, int((top_y + 360) * sy))  # ajusta si el cuadro es más alto
    x1 = int(0.06 * w_orig)
    x2 = int(0.94 * w_orig)

    return (x1, y1, x2, y2), (w_scaled, h_scaled), (w_orig, h_orig)


def ocr_text(img: Image.Image) -> str:
    img = preprocess(img, target_w=TARGET_W)
    return pytesseract.image_to_string(img, lang=LANG, config=CFG_TEXT)


def get_field(text: str, labels, value_pat):
    t = norm(text)
    for lab in labels:
        m = re.search(rf"{lab}\s*[:\-]?\s*{value_pat}", t)
        if m:
            return m.group(1).strip()
    return ""


def best_vin(text: str) -> str:
    t = norm(text)
    cands = re.findall(r"\b[A-Z0-9]{11,20}\b", t)
    if not cands:
        return ""
    cands = sorted(cands, key=lambda x: (abs(len(x) - 17), -len(x)))
    return normalize_code(cands[0])


def extract_fields_from_crop(crop_img: Image.Image) -> dict:
    t = ocr_text(crop_img)

    # Certificado (del header)
    certificado = get_field(
        t,
        labels=[r"CERTI\w* NO\.?", r"CERTIFICADO NO\.?"],
        value_pat=r"([0-9A-Z]{2,10})"
    )

    # Vehículo (bloque derecho); labels tolerantes
    marca = get_field(t, [r"MARCA", r"M4RCA", r"MRCA"], r"([A-Z0-9]{2,25})")
    chasis = get_field(t, [r"CHASIS", r"CH4SIS", r"CHAS1S"], r"([A-Z0-9]{11,20})")
    if not chasis:
        chasis = best_vin(t)
    else:
        chasis = normalize_code(chasis)

    linea = get_field(t, [r"LINEA", r"L1NEA", r"LINFA"], r"([A-Z0-9]{1,12})")

    pasajeros = get_field(t, [r"PASAJER\w*"], r"(\d{1,2})")

    modelo = get_field(t, [r"MODELO", r"M0DELO", r"MODFLO"], r"((?:19|20)\d{2})")
    if not modelo:
        m = re.search(r"\b(19\d{2}|20\d{2})\b", norm(t))
        modelo = m.group(1) if m else ""

    clase = get_field(t, [r"CLASE", r"CL4SE", r"CLASS"], r"([A-Z0-9]{2,20})")
    clase = norm(clase).replace("CANON", "CAMION").replace("CAMOW", "CAMION").replace("CARON", "CAMION")

    placa = get_field(t, [r"PLACA", r"PL4CA", r"PLAC4"], r"([A-Z0-9\-]{2,20})")
    placa = normalize_code(placa)

    motor = get_field(t, [r"MOTOR", r"M0TOR"], r"([A-Z0-9]{2,25})")
    motor = normalize_code(motor)

    color = get_field(t, [r"COLOR", r"C0LOR"], r"([A-Z ]{3,50})")

    # Normalización marca (errores comunes)
    marca = norm(marca).replace("TSUZU", "ISUZU").replace("SUZU", "ISUZU")

    return {
        "certificado_no": norm(certificado),
        "marca": marca,
        "chasis": chasis,
        "linea": norm(linea),
        "pasajeros": norm(pasajeros),
        "modelo": norm(modelo),
        "clase": norm(clase),
        "placa": placa,
        "motor": motor,
        "color": norm(color),
    }


def score(row: dict) -> float:
    # campos clave para tu KPI
    keys = ["certificado_no", "marca", "chasis", "modelo", "clase", "color"]
    got = sum(1 for k in keys if row.get(k))
    return got / len(keys)


def extract_page(pdf_path: str, page_index: int) -> dict:
    last = None
    for zoom in ZOOMS:
        page_img = render_page(pdf_path, page_index, zoom=zoom)

        box, _, _ = find_anchor_box(page_img)
        if box is None:
            # fallback: recorte fijo aproximado del área donde suele estar el cuadro (por si falla ancla)
            w, h = page_img.size
            box = (int(0.06*w), int(0.38*h), int(0.94*w), int(0.74*h))

        crop = page_img.crop(box)
        row = extract_fields_from_crop(crop)
        last = row

        if score(row) >= 0.95:
            break

    return last if last else {
        "certificado_no": "",
        "marca": "",
        "chasis": "",
        "linea": "",
        "pasajeros": "",
        "modelo": "",
        "clase": "",
        "placa": "",
        "motor": "",
        "color": "",
    }


def main():
    pdfs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    if not pdfs:
        print(f"No encontré PDFs en: {INPUT_DIR}")
        return

    results = []
    total_pages = 0
    errors = 0

    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        total_pages += n
        print(f"Procesando {os.path.basename(pdf_path)} ({n} páginas)")

        for i in range(n):
            try:
                results.append(extract_page(pdf_path, i))
            except Exception as e:
                errors += 1
                print(f"  !! ERROR en {os.path.basename(pdf_path)} pág {i+1}: {e}")
                results.append({
                    "certificado_no": "",
                    "marca": "",
                    "chasis": "",
                    "linea": "",
                    "pasajeros": "",
                    "modelo": "",
                    "clase": "",
                    "placa": "",
                    "motor": "",
                    "color": "",
                })

    df = pd.DataFrame(results)
    cols = ["certificado_no","marca","chasis","linea","pasajeros","modelo","clase","placa","motor","color"]
    df = df[cols]

    key_fields = ["certificado_no","marca","chasis","modelo","clase","color"]
    completeness = (
        df[key_fields].astype(str).replace("", pd.NA).notna().sum(axis=1) / len(key_fields)
    ).mean() * 100

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nOK -> {OUTPUT_CSV}")
    print(f"Filas: {len(df)} | Páginas esperadas: {total_pages} | Errores: {errors}")
    print(f"Completitud promedio (campos clave): {completeness:.2f}%")

    # Si querés revisar solo las filas flojas:
    # low = df[(df[key_fields].astype(str).replace('', pd.NA).notna().sum(axis=1) / len(key_fields)) < 0.95]
    # low.to_csv("revisar_bajo_95.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()