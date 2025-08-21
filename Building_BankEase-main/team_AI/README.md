## Project Setup and Installation Guide

This guide will walk you through setting up the project environment and installing the necessary dependencies using `uv`.

## Prerequisites

*   **Git:** Ensure you have Git installed. ([Download Git](https://git-scm.com/downloads))
*   **uv:** Ensure you have `uv` installed. If not, you can install it by following the instructions at [Astral's uv documentation](https://astral.sh/uv/install.sh) or using pipx:
    ```bash
    # Example using pipx (recommended)
    pipx install uv

    # Or using pip (less isolated)
    pip install uv
    ```

## Installation Steps

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone the project repository:
    ```bash
    git clone https://github.com/OmdenaAI/Building_BankEase.git
    cd Building_BankEase/team_AI
    ```

2.  **Create and Activate a Virtual Environment with `uv`:**
    `uv` can create and manage virtual environments. If a `pyproject.toml` file exists, `uv` will use it to understand project dependencies.

    Navigate to the project directory (if you aren't already there) where `pyproject.toml` is located. Then, initialize and activate the virtual environment:

    ```bash
    # This command creates a virtual environment named .venv in the current directory
    # and automatically activates it if your shell is supported.
    # If not automatically activated, follow the manual activation steps below.
    uv venv
    ```

    **Manual Activation (if `uv venv` doesn't auto-activate or if you prefer manual control):**

    *   **Windows (Command Prompt/PowerShell):**
        ```bash
        # If using Command Prompt
        .venv\Scripts\activate.bat

        # If using PowerShell
        .venv\Scripts\Activate.ps1
        ```
        *(Note: On PowerShell, you might need to set the execution policy: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

    *   **Linux / macOS (bash/zsh):**
        ```bash
        source .venv/bin/activate
        ```
    After activation, your terminal prompt should change to indicate that the virtual environment is active (e.g., `(.venv)`).

3.  **Install Dependencies using `uv`:**
    With the virtual environment activated, `uv` will install dependencies defined in your `pyproject.toml` file.

    ```bash
    # This command reads pyproject.toml and installs all dependencies
    # (both main dependencies and development dependencies if specified correctly in pyproject.toml)
    uv pip install -e .
    ```
    *   `-e .`: This installs the current project in "editable" mode. This is standard practice for development, as it means changes to your local project code are immediately reflected in the environment without needing to reinstall. `uv` will look for project metadata (including dependencies) in `pyproject.toml`.

    **Alternatively, to install only runtime dependencies (if that's desired, though `-e .` is common for dev):**
    If your `pyproject.toml` clearly separates runtime and development dependencies, and you only want runtime ones:
    ```bash
    uv pip install .
    ```
    (Without `-e`, it performs a standard installation.)

    **If you also have optional dependency groups defined in `pyproject.toml` (e.g., `[project.optional-dependencies] test = [...]`):**
    ```bash
    # Install main dependencies and the 'test' optional group
    uv pip install -e ".[test]"

    # Install main dependencies and multiple optional groups
    uv pip install -e ".[test,docs]"
    ```

4.  **Verify Installation (Optional):**
    You can list the installed packages to ensure everything is set up:
    ```bash
    uv pip list
    ```

5.  **Deactivating the Virtual Environment:**

    When you are finished working on the project, you can deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Configure Environment Variables
- Create a `.env` file in `team_AI` directory of the project.
- Add your Plaid Sand Box API keys:
```PLAID_SBX_CLIENT_ID=your-plaid-sandbox-client-id
    PLAID_SBX_SECRET=your-plaid-sandbox-client-secret
    PLAID_LINK_USERNAME=plaid-link-user-name
    PLAID_LINK_PASSWORD=plaid-link-password
```
- Adjust configurations in `team_AI/configs/plaid_sbx_configs.yaml` according to your needs:


## Tutorials
- [Setting Up SandBox in Postman](https://www.youtube.com/watch?v=dJds8Qc7weQ)

## Running Plaid Scripts

 ```bash
    - git clone https://github.com/OmdenaAI/Building_BankEase.git
    - cd Building_BankEase/team_AI
    - in "team_AI/configs/plaid_sb_pipeline_configs.yaml"
        - set country in `country_codes`, 
        - Optional: set values in `customers_per_institution`, `institutions_offset`
    
    - Run Pipeline
        - uv run src/plaid_sb_pipeline/main.py     
```


