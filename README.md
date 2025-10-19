# Análise de Jornada de Motoristas

Este repositório contém um conjunto de scripts Python para processar, consolidar e visualizar dados de jornada de motoristas.

## Componentes Principais

### 1. Script de Consolidação de Dados (`merge_data.py`)

Este script é responsável por unificar dados de duas fontes distintas: um sistema de rastreamento (ONIX) e um sistema de relatório de jornada (ATS).

**Funcionalidades:**

*   **Leitura de Dados:** Lê os arquivos `onix.xlsx` e `Relatorio de Jornada.xlsx` localizados no diretório `files/`.
*   **Processamento e Padronização:** Limpa e normaliza os dados, padroniza os nomes das colunas para um formato comum e extrai informações de data e hora.
*   **Consolidação:** Une os dados processados das duas fontes em um único arquivo Excel (`Consolidado.xlsx`), que é salvo no diretório `files/`.
*   **Análise Específica:** Cria uma segunda aba (`Pesquisa Funcionario`) no arquivo consolidado, contendo os horários de início e fim de jornada para um funcionário e data específicos, conforme definido no script.

### 2. Aplicação de Visualização (`streamlit_app.py`)

Uma aplicação web interativa construída com a biblioteca Streamlit para explorar os dados consolidados.

**Funcionalidades:**

*   **Dashboard Interativo:** Apresenta os dados de jornada em uma interface web clara e objetiva.
*   **Filtragem de Dados:** Permite ao usuário filtrar os registros por motorista, data e origem do dado (ONIX ou ATS) através de um painel lateral.
*   **Visualização em Tabela:** Exibe os dados filtrados em uma tabela.
*   **Visualização Geográfica:** Plota as coordenadas de latitude e longitude dos registros em um mapa interativo, permitindo a visualização do trajeto.
*   **Resumo da Pesquisa:** Apresenta o resultado da análise de início e fim de jornada que foi gerada pelo script `merge_data.py`.

## Como Usar

1.  **Preparar os Arquivos:** Certifique-se de que os arquivos `onix.xlsx` e `Relatorio de Jornada.xlsx` estão presentes no diretório `files/`.
2.  **Executar a Consolidação:** Rode o script `merge_data.py` para gerar o arquivo `Consolidado.xlsx` com os dados processados.
    ```bash
    python3 merge_data.py
    ```
3.  **Iniciar a Aplicação Web:** Após a geração do arquivo consolidado, execute o aplicativo Streamlit.
    ```bash
    streamlit run streamlit_app.py
    ```
4.  **Acessar a Aplicação:** Abra o navegador no endereço local fornecido pelo Streamlit para interagir com o dashboard.