import os
import unicodedata

import pandas as pd

BASE_DIR = "/Users/iklo/guto/files"
ONIX_FILE = os.path.join(BASE_DIR, "onix.xlsx")
JORNADA_FILE = os.path.join(BASE_DIR, "Relatorio de Jornada.xlsx")
CONSOLIDATED_FILE = os.path.join(BASE_DIR, "Consolidado.xlsx")

ONIX_SHEET_NAME = "Consulta"
JORNADA_SHEET_NAME = "Relatório de Jornada_teste"
CONSOLIDATED_SHEET_NAME = "Sheet1"

FUNCIONARIO_TESTE = "Accacio"
DATA_TESTE_STR = "04/10/2025"

COMMON_FINAL_COLUMNS = [
    "MOTORISTA",
    "DATA_SEPARADA",
    "HORA",
    "LATITUDE",
    "LONGITUDE",
    "ORIGEM",
    "CIDADE",
    "MACRO",
    "STATUS GERADO",
    "LOCALIZAÇÃO",
]


def normalizar_texto(txt):
    if not isinstance(txt, str):
        return txt
    txt = unicodedata.normalize('NFKD', txt).encode('ASCII', 'ignore').decode('utf-8')
    return txt.strip().upper()


def process_source_dataframe(df, config):
    df = df.drop(columns=config['drop_columns'], errors="ignore")

    df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors="coerce")
    df["DATA_SEPARADA"] = df[config['date_col']].dt.strftime("%d/%m/%Y")
    df["HORA"] = df[config['date_col']].dt.strftime("%H:%M:%S")
    df = df.drop(columns=[config['date_col']])

    df = df.rename(columns=config['rename_columns'])

    df["ORIGEM"] = config['origin_name']

    df = df.reindex(columns=COMMON_FINAL_COLUMNS)
    return df


def filter_employee_data(df, employee_name, target_date):
    df["DATA_SEPARADA"] = pd.to_datetime(df["DATA_SEPARADA"], errors="coerce", dayfirst=True)
    return df[df["MOTORISTA"].astype(str).str.contains(employee_name, case=False, na=False) & (df["DATA_SEPARADA"] == target_date)]


def find_journey_times(df, event_col_name):
    df_copy = df.copy()

    if event_col_name not in df_copy.columns:
        return None, None

    df_copy[event_col_name] = df_copy[event_col_name].astype(str).apply(normalizar_texto)

    hora_inicio = df_copy.loc[df_copy[event_col_name].str.contains("INICIO DE JORNADA", na=False), "HORA"]
    hora_fim = df_copy.loc[df_copy[event_col_name].str.contains("FIM DE JORNADA", na=False), "HORA"]
    return (hora_inicio.iloc[0] if not hora_inicio.empty else None, hora_fim.iloc[0] if not hora_fim.empty else None)


def main():
    """
    This script processes and consolidates driving journey data from two different sources:
    'ONIX' (vehicle tracking system) and 'ATS' (journey report system).
    It then generates a consolidated Excel file and performs a specific search for an employee's
    journey times on a given date, saving the results in a new sheet within the consolidated file.

    The script is designed to be run from the command line.

    Configuration:
        - Input Files: 'onix.xlsx' and 'Relatorio de Jornada.xlsx' are expected in the
          'files' subdirectory relative to where the script is executed.
        - Output File: 'Consolidado.xlsx' will be created/updated in the same 'files' subdirectory.
        - Sheet Names: The script expects specific sheet names within the input Excel files.
        - Test Parameters: 'FUNCIONARIO_TESTE' and 'DATA_TESTE_STR' can be adjusted
          to search for different employees and dates.

    Main Steps:
    1.  **Data Loading and Pre-processing:**
        - Reads data from 'onix.xlsx' and 'Relatorio de Jornada.xlsx'.
        - Cleans and standardizes column names (e.g., 'Condutor' becomes 'MOTORISTA').
        - Extracts separate date and time columns ('DATA_SEPARADA', 'HORA').
        - Adds an 'ORIGEM' column to identify the source of each record ('ONIX' or 'ATS').

    2.  **Data Consolidation:**
        - Vertically combines the processed ONIX and ATS data into a single DataFrame.
        - Missing columns (e.g., 'CIDADE' for ATS records, 'STATUS GERADO' for ONIX records)
          are filled with empty values (NaN).
        - Saves this combined data into 'Consolidado.xlsx' (Sheet1).

    3.  **Employee Journey Search:**
        - Reads the newly created 'Consolidado.xlsx'.
        - Filters the data to find records for a specific employee ('FUNCIONARIO_TESTE')
          on a particular date ('DATA_TESTE_STR').
        - Identifies 'INICIO DE JORNADA' (start of journey) and 'FIM DE JORNADA' (end of journey)
          events from both ONIX (using 'MACRO' column) and ATS (using 'STATUS GERADO' column) data.
        - Compiles these findings into a new DataFrame.
        - Saves this search result into a new sheet named 'Pesquisa Funcionario' within
          the 'Consolidado.xlsx' file.

    4.  **Execution Summary:**
        - Prints a summary at the end, detailing input/output file information,
          record counts, and the results of the employee journey search.

    Usage:
        To run the script, simply execute it from your terminal:
        `python3 merge_data.py`

        Ensure that 'onix.xlsx' and 'Relatorio de Jornada.xlsx' are present in the
        'files' subdirectory before execution."""
    summary_info = {
        "onix_raw_records": 0,
        "jornada_raw_records": 0,
        "consolidated_records": 0,
        "employee_search_performed": False,
        "employee_name_searched": FUNCIONARIO_TESTE,
        "date_searched": DATA_TESTE_STR,
        "onix_journey_found": False,
        "ats_journey_found": False,
        "pesquisa_df_content": "Nenhum dado encontrado.",
    }

    print("--- Iniciando processamento e consolidação dos arquivos ---")

    onix_config = {
        'sheet_name': ONIX_SHEET_NAME,
        'drop_columns': ["PLACA", "VELOCI"],
        'date_col': "DATA",
        'rename_columns': {"MOTORISTA": "MOTORISTA", "CIDADE": "CIDADE", "LATITU": "LATITUDE", "LONGIT": "LONGITUDE", "MACRO": "MACRO"},
        'origin_name': "ONIX",
    }

    jornada_config = {
        'sheet_name': JORNADA_SHEET_NAME,
        'drop_columns': ["Usuário", "Data Geração", "Identificador 2"],
        'date_col': "Data Cadastro",
        'rename_columns': {
            "Condutor": "MOTORISTA",
            "Status Gerado": "STATUS GERADO",
            "Localização": "LOCALIZAÇÃO",
            "Latitude": "LATITUDE",
            "Longitude": "LONGITUDE",
        },
        'origin_name': "ATS",
    }

    try:
        onix_raw_df = pd.read_excel(ONIX_FILE, sheet_name=onix_config['sheet_name'])
        summary_info["onix_raw_records"] = len(onix_raw_df)
        onix_processed_df = process_source_dataframe(onix_raw_df, onix_config)

        jornada_raw_df = pd.read_excel(JORNADA_FILE, sheet_name=jornada_config['sheet_name'])
        summary_info["jornada_raw_records"] = len(jornada_raw_df)
        jornada_processed_df = process_source_dataframe(jornada_raw_df, jornada_config)

        consolidated_df = pd.concat([onix_processed_df, jornada_processed_df], ignore_index=True)
        summary_info["consolidated_records"] = len(consolidated_df)

        consolidated_df.to_excel(CONSOLIDATED_FILE, index=False)
        print(f"\n✅ Planilha consolidada criada com sucesso em:\n{CONSOLIDATED_FILE}")

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        return
    except Exception as e:
        print(f"Ocorreu um erro durante a consolidação: {e}")
        return

    print("\n--- Iniciando processamento do arquivo consolidado para pesquisa de funcionário ---")
    summary_info["employee_search_performed"] = True

    try:
        df_consolidated = pd.read_excel(CONSOLIDATED_FILE, sheet_name=CONSOLIDATED_SHEET_NAME)
        df_consolidated.columns = [normalizar_texto(c) for c in df_consolidated.columns]
        print("✅ Colunas encontradas no arquivo consolidado:", list(df_consolidated.columns))

        onix_data_for_search = df_consolidated[df_consolidated['ORIGEM'] == 'ONIX'].copy()
        ats_data_for_search = df_consolidated[df_consolidated['ORIGEM'] == 'ATS'].copy()

        target_date = pd.to_datetime(DATA_TESTE_STR, dayfirst=True)

        onix_filtered = filter_employee_data(onix_data_for_search, FUNCIONARIO_TESTE, target_date)
        ats_filtered = filter_employee_data(ats_data_for_search, FUNCIONARIO_TESTE, target_date)

        hora_onix_ini, hora_onix_fim = find_journey_times(onix_filtered, "MACRO")
        hora_ats_ini, hora_ats_fim = find_journey_times(ats_filtered, "STATUS GERADO")

        if hora_onix_ini or hora_onix_fim:
            summary_info["onix_journey_found"] = True
        if hora_ats_ini or hora_ats_fim:
            summary_info["ats_journey_found"] = True

        result_data = []
        data_fmt = target_date.strftime("%d/%m/%Y")

        if hora_onix_ini or hora_onix_fim:
            result_data.append(
                {
                    "FONTE": "ONIX",
                    "MOTORISTA": FUNCIONARIO_TESTE,
                    "DATA": data_fmt,
                    "HORA INÍCIO VIAGEM": hora_onix_ini,
                    "FIM DA VIAGEM": hora_onix_fim,
                }
            )

        if hora_ats_ini or hora_ats_fim:
            result_data.append(
                {
                    "FONTE": "ATS",
                    "MOTORISTA": FUNCIONARIO_TESTE,
                    "DATA": data_fmt,
                    "HORA INÍCIO VIAGEM": hora_ats_ini,
                    "FIM DA VIAGEM": hora_ats_fim,
                }
            )

        pesquisa_df = pd.DataFrame(result_data)

        if not pesquisa_df.empty:
            summary_info["pesquisa_df_content"] = pesquisa_df.to_string(index=False)
        else:
            summary_info["pesquisa_df_content"] = "Nenhum dado de jornada encontrado para o funcionário e data especificados."

        with pd.ExcelWriter(CONSOLIDATED_FILE, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            pesquisa_df.to_excel(writer, sheet_name="Pesquisa Funcionario", index=False)

        print(f"✅ Aba 'Pesquisa Funcionario' criada com sucesso para {FUNCIONARIO_TESTE} em {data_fmt}")
        print(" Dados encontrados:")
        print(pesquisa_df)

    except FileNotFoundError as e:
        print(f"Erro: Arquivo consolidado não encontrado - {e}")
    except Exception as e:
        print(f"Ocorreu um erro durante a pesquisa de funcionário: {e}")

    print("\n" + "=" * 50)
    print("                 RESUMO DA EXECUÇÃO")
    print("=" * 50)
    print("Arquivos de Entrada:")
    print(f"  ONIX: {ONIX_FILE} ({summary_info['onix_raw_records']} registros)")
    print(f"  Jornada: {JORNADA_FILE} ({summary_info['jornada_raw_records']} registros)")
    print("\nArquivo Consolidado Gerado:")
    print(f"  Caminho: {CONSOLIDATED_FILE}")
    print(f"  Total de Registros: {summary_info['consolidated_records']}")
    print("\nPesquisa de Funcionário:")
    print(f"  Funcionário: {summary_info['employee_name_searched']}")
    print(f"  Data: {summary_info['date_searched']}")
    print(f"  Jornada ONIX encontrada: {'Sim' if summary_info['onix_journey_found'] else 'Não'}")
    print(f"  Jornada ATS encontrada: {'Sim' if summary_info['ats_journey_found'] else 'Não'}")
    print("\nConteúdo da Aba 'Pesquisa Funcionario':")
    print(summary_info['pesquisa_df_content'])
    print("=" * 50)


if __name__ == "__main__":
    main()
