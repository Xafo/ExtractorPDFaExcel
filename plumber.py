import os
import re
import glob
import fitz
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# =========================
# Tesseract path
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# CONFIG
# =========================
INPUT_DIR = r"./pdfs"
OUTPUT_CSV = r"./certificados.csv"
LANG = "spa"
PAGE_ZOOM = 4.0

# Recorte amplio: desde "DATOS GENERALES" hasta después de "Color"
# Probamos múltiples y nos quedamos con el mejor
BOX_CANDIDATES = [
    (0.0, 0.20, 1.0, 0.60),   # ultra amplio (fallback)
    (0.02, 0.24, 0.98, 0.50),
    (0.02, 0.27, 0.98, 0.53),
]


def preprocess(img):
    img = img.convert("L")
    target_w = 3200
    w, h = img.size
    if w != target_w:
        s = target_w / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def render_page(pdf_path, page_index):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(PAGE_ZOOM, PAGE_ZOOM), alpha=False)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_rel(page_img, box):
    x1, y1, x2, y2 = box
    w, h = page_img.size
    return page_img.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))


# =========================
# ESTRATEGIA: word-level OCR con image_to_data
# Encontrar labels y leer los valores a su derecha
# =========================

LABEL_MAP = {
    "certificado":  {"search": ["certificado"], "next_after": ["no", "no."], "field": "certificado_no"},
    "marca":        {"search": ["marca"],       "field": "marca"},
    "chasis":       {"search": ["chasis"],      "field": "chasis"},
    "linea":        {"search": ["linea", "línea"], "field": "linea"},
    "pasajeros":    {"search": ["pasajeros"],   "field": "pasajeros"},
    "modelo":       {"search": ["modelo"],      "field": "modelo"},
    "clase":        {"search": ["clase"],       "field": "clase"},
    "placa":        {"search": ["placa"],       "field": "placa"},
    "motor":        {"search": ["motor"],       "field": "motor"},
    "color":        {"search": ["color"],       "field": "color"},
}


def find_value_after_label(words, label_idx, all_words):
    """
    Dado un label encontrado en words[label_idx], busca el valor
    que está a su derecha o inmediatamente después (con ':' en medio).
    
    words es una lista de dicts con: text, left, top, width, height
    """
    label = all_words[label_idx]
    label_right = label["left"] + label["width"]
    label_top = label["top"]
    label_bottom = label_top + label["height"]
    label_center_y = label_top + label["height"] / 2
    
    # Buscar palabras que estén:
    # 1. A la derecha del label (o del ":" después del label)
    # 2. En la misma línea vertical (similar Y)
    
    # Primero, buscar si hay un ":" justo después
    colon_right = label_right
    for w in all_words:
        if w["text"] in [":", ".:"] and abs(w["top"] - label_top) < label["height"] * 0.5:
            if w["left"] > label_right - 10 and w["left"] < label_right + label["width"] * 2:
                colon_right = w["left"] + w["width"]
                break
    
    # Buscar todas las palabras a la derecha del label/colon en la misma línea
    value_words = []
    y_tolerance = label["height"] * 0.8
    
    for w in all_words:
        # Misma línea vertical
        w_center_y = w["top"] + w["height"] / 2
        if abs(w_center_y - label_center_y) > y_tolerance:
            continue
        # A la derecha
        if w["left"] <= colon_right:
            continue
        # No demasiado lejos
        if w["left"] > label_right + 1500:
            continue
        # Ignorar otros labels conocidos
        if w["text"].lower().rstrip(":.,") in ["marca", "chasis", "linea", "línea", 
            "pasajeros", "modelo", "clase", "placa", "motor", "color",
            "descripción", "descripcion", "del", "vehículo", "vehiculo",
            "forma", "pago", "código", "codigo", "agente", "nombre"]:
            continue
        value_words.append(w)
    
    # Ordenar por posición X
    value_words.sort(key=lambda w: w["left"])
    
    # Concatenar
    if value_words:
        return " ".join(w["text"] for w in value_words).strip()
    return ""


def extract_with_spatial(img):
    """Extrae campos usando OCR word-level con posiciones espaciales."""
    processed = preprocess(img)
    
    # Obtener datos word-level
    data = pytesseract.image_to_data(processed, lang=LANG, 
                                      config="--oem 1 --psm 6",
                                      output_type=pytesseract.Output.DICT)
    
    # Construir lista de palabras con posiciones
    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 10:  # ignorar basura
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf": conf,
            "idx": i,
        })
    
    # Para cada campo, buscar su label y extraer el valor
    results = {}
    
    for field_name, info in LABEL_MAP.items():
        search_terms = info["search"]
        field_key = info["field"]
        
        best_value = ""
        
        for si, w in enumerate(words):
            w_text = w["text"].lower().rstrip(":.,;")
            
            if w_text in search_terms:
                # Para "Certificado No.:", necesitamos saltar "No." 
                if field_name == "certificado":
                    # Buscar "No" o "No." después de "Certificado"
                    # El valor numérico viene después
                    val = find_value_after_label(words, si, words)
                    # Buscar el número en el valor
                    # El valor puede ser "No.: 444" o similar
                    nums = re.findall(r"\d{1,4}", val)
                    # Filtrar: queremos el número de certificado (3 dígitos típicamente)
                    for num in nums:
                        n_val = int(num)
                        if 1 <= n_val <= 9999:
                            best_value = num
                            break
                else:
                    val = find_value_after_label(words, si, words)
                    # Limpiar ":" al inicio
                    val = val.lstrip(":").strip()
                    if val and len(val) > len(best_value):
                        best_value = val
        
        results[field_key] = best_value
    
    return results


def clean_results(row):
    """Limpia y normaliza los valores extraídos."""
    # Certificado
    cert = row.get("certificado_no", "")
    m = re.search(r"(\d{1,4})", cert)
    row["certificado_no"] = m.group(1) if m else ""
    
    # Marca
    marca = row.get("marca", "").upper().strip()
    marca = re.sub(r"[^A-Z]", "", marca)
    fixes = {"IISUZU": "ISUZU", "TSUZU": "ISUZU", "LSUZU": "ISUZU",
             "1SUZU": "ISUZU", "SUZU": "ISUZU", "ISUZUI": "ISUZU",
             "ISUSU": "ISUZU"}
    row["marca"] = fixes.get(marca, marca)
    
    # Chasis
    chasis = row.get("chasis", "").upper()
    chasis = re.sub(r"[^A-Z0-9]", "", chasis)
    # Fix OCR común: en el patrón JAANPR71H, "I" puede ser "1" y viceversa
    # No hacer fix global, solo validar longitud
    row["chasis"] = chasis if 11 <= len(chasis) <= 20 else ""
    
    # Línea
    linea = row.get("linea", "").upper()
    linea = re.sub(r"[^A-Z0-9]", "", linea)
    row["linea"] = linea
    
    # Pasajeros
    pas = row.get("pasajeros", "")
    m = re.search(r"(\d{1,2})", pas)
    row["pasajeros"] = m.group(1) if m and 1 <= int(m.group(1)) <= 60 else ""
    
    # Modelo
    mod = row.get("modelo", "")
    m = re.search(r"((?:19|20)\d{2})", mod)
    row["modelo"] = m.group(1) if m else ""
    
    # Clase
    clase = row.get("clase", "").upper()
    clase = re.sub(r"[^A-Z]", "", clase)
    fixes_c = {"CANON": "CAMION", "CAMON": "CAMION", "CAMWON": "CAMION",
               "CANION": "CAMION", "CAM1ON": "CAMION", "CAMIDN": "CAMION",
               "CAMLON": "CAMION"}
    row["clase"] = fixes_c.get(clase, clase)
    
    # Placa
    placa = row.get("placa", "").upper()
    placa = placa.replace(" ", "").replace(".", "")
    placa = re.sub(r"[^A-Z0-9\-]", "", placa)
    row["placa"] = placa if 3 <= len(placa) <= 15 else ""
    
    # Motor
    motor = row.get("motor", "").upper()
    motor = re.sub(r"[^A-Z0-9]", "", motor)
    row["motor"] = motor if 3 <= len(motor) <= 15 else ""
    
    # Color
    color = row.get("color", "").upper()
    color = re.sub(r"[^A-Z]", "", color)
    row["color"] = color if len(color) >= 3 else ""
    
    return row


# =========================
# TAMBIÉN intentamos regex sobre texto plano como fallback
# =========================
def extract_with_regex(img):
    processed = preprocess(img)
    # Intentar múltiples PSM modes
    results = {}
    
    for psm in [6, 11, 4]:
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        raw = pytesseract.image_to_string(processed, lang=LANG, config=cfg)
        raw = raw.replace("\x0c", "")
        text = re.sub(r"[ \t]+", " ", raw)
        
        def ef(pattern):
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(1).strip() if m else ""
        
        fields = {
            "certificado_no": ef(r"Certificado\s*(?:No|N[o0])\.?\s*:?\s*(\d{1,4})"),
            "marca": ef(r"Marca\s*:?\s*([A-Za-z]{3,10})"),
            "chasis": ef(r"(?:Chasis|Chassis)\s*:?\s*([A-Z0-9]{10,20})"),
            "linea": ef(r"L[ií]nea\s*:?\s*([A-Za-z0-9]{1,10})"),
            "pasajeros": ef(r"Pasajeros\s*:?\s*(\d{1,2})"),
            "modelo": ef(r"Modelo\s*:?\s*((?:19|20)\d{2})"),
            "clase": ef(r"Clase\s*:?\s*([A-Za-z]{4,15})"),
            "placa": ef(r"Placa\s*:?\s*([A-Za-z0-9\s\-\.]{3,15})"),
            "motor": ef(r"Motor\s*:?\s*([A-Za-z0-9]{3,10})"),
            "color": ef(r"Color\s*:?\s*([A-Za-z]{4,15})"),
        }
        
        # Merge: guardar el campo si no lo teníamos
        for k, v in fields.items():
            if v and not results.get(k):
                results[k] = v
    
    return results


def merge_results(spatial, regex):
    """Combina resultados de ambos métodos, priorizando el que tenga valor."""
    merged = {}
    all_keys = ["certificado_no", "marca", "chasis", "linea", "pasajeros",
                "modelo", "clase", "placa", "motor", "color"]
    
    for k in all_keys:
        s_val = spatial.get(k, "")
        r_val = regex.get(k, "")
        
        # Preferir el valor más largo/completo
        if s_val and r_val:
            merged[k] = s_val if len(s_val) >= len(r_val) else r_val
        else:
            merged[k] = s_val or r_val
    
    return merged


def extract_page(pdf_path, page_index):
    page_img = render_page(pdf_path, page_index)
    
    best_row = None
    best_score = -1
    
    for box in BOX_CANDIDATES:
        section = crop_rel(page_img, box)
        
        # Método 1: Espacial (word-level)
        spatial = extract_with_spatial(section)
        spatial = clean_results(spatial)
        
        # Método 2: Regex sobre texto plano
        regex_res = extract_with_regex(section)
        regex_res = clean_results(regex_res)
        
        # Combinar
        merged = merge_results(spatial, regex_res)
        
        # Score
        keys = ["certificado_no", "marca", "chasis", "modelo", "clase", "color"]
        sc = sum(1 for k in keys if merged.get(k))
        
        if sc > best_score:
            best_score = sc
            best_row = merged
        
        if sc >= 5:
            break
    
    return best_row or {k: "" for k in ["certificado_no", "marca", "chasis", "linea",
                                         "pasajeros", "modelo", "clase", "placa", "motor", "color"]}


def main():
    pdfs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    if not pdfs:
        print(f"No encontré PDFs en: {INPUT_DIR}")
        return

    results = []
    total_pages = 0

    for pdf in pdfs:
        doc = fitz.open(pdf)
        n = len(doc)
        doc.close()
        total_pages += n
        print(f"Procesando {os.path.basename(pdf)} ({n} páginas)")

        for i in range(n):
            row = extract_page(pdf, i)
            results.append(row)
            filled = sum(1 for v in row.values() if v)
            print(f"  Pág {i+1:2d}: cert={row['certificado_no']:>3s} marca={row['marca']:<6s} "
                  f"chasis={row['chasis']:<17s} linea={row['linea']:<4s} "
                  f"pas={row['pasajeros']:<2s} modelo={row['modelo']:<4s} "
                  f"clase={row['clase']:<7s} placa={row['placa']:<10s} "
                  f"motor={row['motor']:<8s} color={row['color']:<7s} "
                  f"({filled}/10)")

    df = pd.DataFrame(results)
    cols = ["certificado_no", "marca", "chasis", "linea", "pasajeros",
            "modelo", "clase", "placa", "motor", "color"]
    df = df[cols]

    key_fields = ["certificado_no", "marca", "chasis", "modelo", "clase", "color"]
    completeness = (
        df[key_fields].astype(str).replace("", pd.NA).notna().sum(axis=1) / len(key_fields)
    ).mean() * 100

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nOK -> {OUTPUT_CSV}")
    print(f"Filas: {len(df)} | Páginas esperadas: {total_pages}")
    print(f"Completitud promedio (campos clave): {completeness:.2f}%")


if __name__ == "__main__":
    main()