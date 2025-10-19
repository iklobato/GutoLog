# Análise de Jornada de Motoristas

Este repositório contém ferramentas para processar, consolidar e visualizar dados de jornada de motoristas a partir de duas fontes de dados: um sistema de rastreamento (ONIX) e um relatório de jornada (ATS).

## Estrutura do Projeto

*   `app.py`: A aplicação principal, um dashboard interativo e completo construído com Streamlit.
*   `merge_data.py`: Um script de linha de comando para consolidar os dados de forma não-interativa.
*   `requirements.txt`: Lista de dependências Python para o projeto.
*   `files/`: Diretório destinado a armazenar os arquivos de dados de entrada e saída.

---

## Como Utilizar

Existem duas maneiras de usar este projeto.

### Método 1: Dashboard Interativo (Recomendado)

Esta é a forma mais completa e amigável de utilizar a ferramenta. A aplicação `app.py` permite fazer o upload dos arquivos, processar os dados e visualizar as análises em um só lugar, sem a necessidade de executar scripts separados.

**Passos:**

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```

3.  **Use a Interface:**
    *   No painel lateral da aplicação, faça o upload dos seus arquivos `onix.xlsx` e `Relatorio de Jornada.xlsx`.
    *   Crie e carregue uma nova análise para ver os dados processados e visualizações interativas.

### Método 2: Script de Linha de Comando

Este método é ideal para automação ou para quem prefere um fluxo de trabalho via terminal. O script `merge_data.py` consolida os arquivos de dados.

**Passos:**

1.  **Adicione os arquivos de dados:**
    *   Coloque os arquivos `onix.xlsx` e `Relatorio de Jornada.xlsx` dentro da pasta `files/`.

2.  **Execute o script de consolidação:**
    ```bash
    python3 merge_data.py
    ```

3.  **Verifique o resultado:**
    *   O script irá criar o arquivo `Consolidado.xlsx` dentro da pasta `files/`, contendo os dados unificados e uma aba de análise.
