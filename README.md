# Freight Budget Management

## Estrutura do Projeto

*   `app.py`: A aplicação principal construída com Streamlit.
*   `requirements.txt`: Lista de dependências Python para o projeto.
*   `freight_budget_management/`: Módulo com regras de domínio, serviços e UI.

---

## Como Utilizar

Esta é a forma recomendada de utilizar a ferramenta.

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
    *   A aplicação abre diretamente no fluxo de Freight Budget Management.

*   Criar cotações com itens de linha e validade.
*   Executar comandos de ciclo de vida conforme a especificação.
*   Gerar PDFs padronizados (com marcação “Draft/Unapproved” quando aplicável).
*   Consultar métricas de cotações e exportar CSV.
*   Visualizar a lista de cotações, aplicar filtros e editar registros com bloqueio de edição.
*   Calcular cotações com o motor de regras (freight quote engine).
