"""
Extractor de datos de certificados de pólizas vehiculares (PDF escaneado)
Usa OCR (pytesseract + pdf2image) para PDFs basados en imágenes.
Exporta resultados a Excel (.xlsx) y CSV.

Dependencias:
    pip install pytesseract pdf2image pandas openpyxl

Requisitos del sistema:
    - tesseract-ocr (apt install tesseract-ocr)
    - poppler-utils (apt install poppler-utils)

Uso:
    python extraer_polizas.py                            # Procesa todos los PDFs en ./pdfs/
    python extraer_polizas.py -i archivo.pdf             # Procesa un solo archivo
    python extraer_polizas.py -i carpeta_pdfs/           # Procesa carpeta
    python extraer_polizas.py -i carpeta/ -o resultado   # Nombre de salida personalizado
    python extraer_polizas.py -i archivo.pdf --dpi 400   # Mayor DPI para mejor OCR
"""

import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import re
import os
import sys
import argparse
from pathlib import Path


# ============================================================
# CONFIGURACIÓN - Ajustar según formato de tus pólizas
# ============================================================
DPI_DEFAULT = 300
TESSERACT_LANG = "eng"
MARCADOR_PAGINA = "DATOS GENERALES"


# ============================================================
# FUNCIONES DE EXTRACCIÓN
# ============================================================

def extraer_campo(texto, patrones, grupo=1):
    """Intenta múltiples patrones regex para extraer un campo."""
    if isinstance(patrones, str):
        patrones = [patrones]
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            valor = match.group(grupo).strip()
            return " ".join(valor.split())
    return None


def extraer_montos(texto):
    """Extrae Suma Asegurada y Deducible de la sección de coberturas."""
    resultado = {
        "Suma Asegurada (Todo Riesgo excl. robo)": None,
        "Deducible (Todo Riesgo excl. robo)": None,
        "Suma Asegurada (Robo y/o Hurto Total)": None,
        "Deducible (Robo y/o Hurto Total)": None,
    }

    seccion = re.search(
        r"(?:Todo Riesgo excluyendo robo|excluyendo robo)(.*)",
        texto, re.IGNORECASE | re.DOTALL
    )
    if not seccion:
        return resultado

    bloque = seccion.group(1)
    montos_raw = re.findall(r"([\d,]+\.\d{2})", bloque)

    montos = []
    for m in montos_raw:
        try:
            montos.append(float(m.replace(",", "")))
        except ValueError:
            continue

    if len(montos) >= 4:
        resultado["Suma Asegurada (Todo Riesgo excl. robo)"] = montos[0]
        resultado["Deducible (Todo Riesgo excl. robo)"] = montos[1]
        resultado["Suma Asegurada (Robo y/o Hurto Total)"] = montos[2]
        resultado["Deducible (Robo y/o Hurto Total)"] = montos[3]
    elif len(montos) == 2:
        resultado["Suma Asegurada (Todo Riesgo excl. robo)"] = montos[0]
        resultado["Suma Asegurada (Robo y/o Hurto Total)"] = montos[1]
    elif len(montos) == 1:
        resultado["Suma Asegurada (Todo Riesgo excl. robo)"] = montos[0]

    return resultado


def extraer_certificado(texto_pagina):
    """Extrae todos los campos de un certificado desde texto OCR."""
    if not texto_pagina or MARCADOR_PAGINA not in texto_pagina:
        return None

    d = {}

    # Datos de la póliza
    d["Ramo"] = extraer_campo(texto_pagina, r"Ramo:\s*(\w+)")
    d["Tipo de Cobertura"] = extraer_campo(texto_pagina, [
        r"Tipo de cobertura:\s*(.+?)(?:\s{2,}|P[oó]liza)",
        r"cobertura:\s*(.+?)(?:\s{2,}|Poliza)",
    ])
    d["No. Póliza"] = extraer_campo(texto_pagina, r"P[oó]liza No\.?:\s*(.+?)(?:\s{2,}|Certificado|\n)")
    d["No. Certificado"] = extraer_campo(texto_pagina, r"Certificado No\.?:?\s*(\d+)")
    d["Inicio Vigencia"] = extraer_campo(texto_pagina, r"Inicio de Vigencia:\s*([\d/]+)")
    d["Fin Vigencia"] = extraer_campo(texto_pagina, r"Fin de Vigencia:\s*([\d/]+)")

    # Asegurado
    d["Asegurado"] = extraer_campo(texto_pagina, r"Asegurado:\s*(.+?)(?:\s{2,}|Descripci[oó]n|\n)")
    if d["Asegurado"] and "SOCIEDAD" in d["Asegurado"] and "ANONIMA" not in d["Asegurado"]:
        d["Asegurado"] += " ANONIMA"

    d["Propietario"] = extraer_campo(texto_pagina, r"Propietario:\s*(.+?)(?:\s{2,}|L[ií]nea|\n)")
    if d["Propietario"] and "SOCIEDAD" in d["Propietario"] and "ANONIMA" not in d["Propietario"]:
        d["Propietario"] += " ANONIMA"

    d["NIT"] = extraer_campo(texto_pagina, r"Nit:\s*(\d+)")
    d["Dirección"] = extraer_campo(texto_pagina, r"Direcci[oó]n:\s*(.+?)(?:\n|Color)")

    # Vehículo
    d["Marca"] = extraer_campo(texto_pagina, r"Marca:\s*([A-Z!]\w*)")
    if d["Marca"]:
        correcciones = {"!SUZU": "ISUZU", "ISUZU": "ISUZU", "!suzu": "ISUZU"}
        d["Marca"] = correcciones.get(d["Marca"], d["Marca"])

    d["Línea"] = extraer_campo(texto_pagina, [
        r"L[ií]nea:\s*(.+?)(?:\s{2,}|Pasajeros|\n)",
    ])
    d["Modelo (Año)"] = extraer_campo(texto_pagina, r"Modelo:\s*(\d{4})")
    d["Clase"] = extraer_campo(texto_pagina, [
        r"Clase:\s*(.+?)(?:\s{2,}|\n)",
    ])
    d["Pasajeros"] = extraer_campo(texto_pagina, r"Pasajeros:\s*(\d+)")
    d["Placa"] = extraer_campo(texto_pagina, r"Placa:\s*([A-Z0-9][\s\-]*[\d\w]+)")
    if d["Placa"]:
        d["Placa"] = d["Placa"].replace(" ", "")

    d["Chasis"] = extraer_campo(texto_pagina, r"Chasis:\s*(\S+)")
    d["Motor"] = extraer_campo(texto_pagina, r"Motor:\s*(.+?)(?:\s{2,}|\n)")
    d["Color"] = extraer_campo(texto_pagina, r"Color:\s*(.+?)(?:\s{2,}|\n)")

    # Agente
    d["Código Agente"] = extraer_campo(texto_pagina, r"[Cc][oó]digo del agente:?\s*(\d+)")
    d["Nombre Agente"] = extraer_campo(texto_pagina, [
        r"Nombre del agente:?\s*(.+?)(?:\n|$)",
        r"Nombre d\w+ agente:?\s*(.+?)(?:\n|$)",
    ])
    d["Forma de Pago"] = extraer_campo(texto_pagina, r"Forma de Pago:\s*(\w+)")
    d["Moneda"] = extraer_campo(texto_pagina, r"Moneda:\s*(\w+)")

    # Montos
    montos = extraer_montos(texto_pagina)
    d.update(montos)

    return d


# ============================================================
# PROCESAMIENTO DE ARCHIVOS
# ============================================================

def procesar_pdf(ruta_pdf, dpi=DPI_DEFAULT):
    """Procesa un PDF con OCR y retorna lista de certificados."""
    certificados = []
    nombre = os.path.basename(ruta_pdf)
    print(f"  Procesando: {nombre}")

    try:
        print(f"    Convirtiendo a imágenes (DPI={dpi})...")
        imagenes = convert_from_path(ruta_pdf, dpi=dpi)
        print(f"    {len(imagenes)} páginas. Ejecutando OCR...")

        for i, img in enumerate(imagenes):
            texto = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
            datos = extraer_certificado(texto)
            if datos:
                datos["Archivo Origen"] = nombre
                datos["Página"] = i + 1
                certificados.append(datos)
                cert = datos.get("No. Certificado", "?")
                marca = datos.get("Marca", "?")
                placa = datos.get("Placa", "?")
                print(f"      Pág {i+1}: Cert #{cert} | {marca} | {placa}")

        print(f"    -> {len(certificados)} certificados extraídos")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    return certificados


def buscar_pdfs(ruta):
    """Busca archivos PDF."""
    p = Path(ruta)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [str(p)]
    elif p.is_dir():
        return sorted([str(f) for f in p.glob("*.pdf")])
    print(f"Error: '{ruta}' no es un PDF ni directorio válido.")
    return []


def exportar_resultados(certificados, nombre_salida="polizas_extraidas"):
    """Exporta a Excel y CSV."""
    if not certificados:
        print("\nNo se encontraron certificados.")
        return None

    df = pd.DataFrame(certificados)

    columnas_orden = [
        "Archivo Origen", "Página", "No. Póliza", "No. Certificado",
        "Ramo", "Tipo de Cobertura", "Inicio Vigencia", "Fin Vigencia",
        "Asegurado", "Propietario", "NIT", "Dirección",
        "Marca", "Línea", "Modelo (Año)", "Clase", "Pasajeros",
        "Placa", "Chasis", "Motor", "Color",
        "Suma Asegurada (Todo Riesgo excl. robo)", "Deducible (Todo Riesgo excl. robo)",
        "Suma Asegurada (Robo y/o Hurto Total)", "Deducible (Robo y/o Hurto Total)",
        "Código Agente", "Nombre Agente", "Forma de Pago", "Moneda"
    ]
    cols = [c for c in columnas_orden if c in df.columns]
    cols += [c for c in df.columns if c not in columnas_orden]
    df = df[cols]

    xlsx = f"{nombre_salida}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Certificados")
        ws = writer.sheets["Certificados"]
        for i, col in enumerate(df.columns, 1):
            w = max(len(str(col)), df[col].astype(str).str.len().max() if len(df) else 0)
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(w + 2, 45)
    print(f"\n  Excel: {xlsx}")

    csv_path = f"{nombre_salida}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV:   {csv_path}")

    print(f"\n{'='*50}")
    print(f"  Certificados: {len(df)}")
    print(f"  Archivos:     {df['Archivo Origen'].nunique()}")
    if "Marca" in df.columns:
        print(f"  Marcas:       {', '.join(df['Marca'].dropna().unique())}")
    if "Suma Asegurada (Todo Riesgo excl. robo)" in df.columns:
        print(f"  Suma Total:   Q {df['Suma Asegurada (Todo Riesgo excl. robo)'].sum():,.2f}")
    print(f"{'='*50}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Extractor de pólizas vehiculares (OCR)")
    parser.add_argument("-i", "--input", default="./pdfs", help="PDF o directorio (default: ./pdfs)")
    parser.add_argument("-o", "--output", default="polizas_extraidas", help="Nombre salida (default: polizas_extraidas)")
    parser.add_argument("--dpi", type=int, default=DPI_DEFAULT, help=f"DPI (default: {DPI_DEFAULT})")
    args = parser.parse_args()

    print("=" * 60)
    print("  EXTRACTOR DE PÓLIZAS VEHICULARES (OCR)")
    print("=" * 60)

    archivos = buscar_pdfs(args.input)
    if not archivos:
        sys.exit(1)

    print(f"\nArchivos PDF: {len(archivos)}\n")
    todos = []
    for pdf in archivos:
        todos.extend(procesar_pdf(pdf, dpi=args.dpi))

    exportar_resultados(todos, args.output)


if __name__ == "__main__":
    main()