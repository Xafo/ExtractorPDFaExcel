import pdfplumber
import csv
import os
import re

OUTPUT_FILE = "vehicles.csv"
PDF_FILES = [f"{i}.pdf" for i in range(1, 11)]

def clean_value(text):
    """Limpia el valor extraído eliminando texto sobrante."""
    # Elimina saltos de línea y espacios extra
    return text.strip().replace('\n', ' ')

def extract_field(text, field_name):
    """Extrae el valor de un campo específico del texto."""
    # Busca el patrón: "NombreCampo: VALOR" hasta el siguiente campo o salto
    pattern = rf"{re.escape(field_name)}:\s*([^\n]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return clean_value(match.group(1))
    return ""

def extract_page_data(page, pdf_filename, page_num):
    """Extrae datos de vehículo de una página."""
    text = page.extract_text()
    if not text or "Descripción del Vehículo" not in text and "Descripcion del Vehiculo" not in text:
        return None

    # Extraer número de certificado
    cert_match = re.search(r"Certificado No\.?:\s*(\S+)", text, re.IGNORECASE)
    certificado = cert_match.group(1) if cert_match else ""

    # Extraer número de póliza
    poliza_match = re.search(r"Póliza No\.?:\s*(\S+)", text, re.IGNORECASE)
    poliza = poliza_match.group(1) if poliza_match else ""

    return {
        "Archivo": pdf_filename,
        "Pagina": page_num,
        "Poliza": poliza,
        "Certificado": certificado,
        "Marca": extract_field(text, "Marca"),
        "Chasis": extract_field(text, "Chasis"),
        "Linea": extract_field(text, "Línea") or extract_field(text, "Linea"),
        "Pasajeros": extract_field(text, "Pasajeros"),
        "Modelo": extract_field(text, "Modelo"),
        "Clase": extract_field(text, "Clase"),
        "Placa": extract_field(text, "Placa"),
        "Motor": extract_field(text, "Motor"),
        "Color": extract_field(text, "Color"),
    }

all_vehicles = []

for pdf_file in PDF_FILES:
    if not os.path.exists(pdf_file):
        print(f"[INFO] Archivo no encontrado, omitiendo: {pdf_file}")
        continue

    print(f"[INFO] Procesando: {pdf_file}")

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            data = extract_page_data(page, pdf_file, page_num)
            if data:
                all_vehicles.append(data)
                print(f"  -> Página {page_num}: Cert. {data['Certificado']} | {data['Marca']} {data['Linea']} | Chasis: {data['Chasis']}")
            else:
                print(f"  -> Página {page_num}: sin datos de vehículo")

if all_vehicles:
    fieldnames = ["Archivo", "Pagina", "Poliza", "Certificado", "Marca", "Chasis", "Linea", "Pasajeros", "Modelo", "Clase", "Placa", "Motor", "Color"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_vehicles)
    print(f"\n[DONE] CSV creado: {OUTPUT_FILE} — {len(all_vehicles)} vehículos extraídos.")
else:
    print("\n[ERROR] No se encontraron datos en ningún PDF.")