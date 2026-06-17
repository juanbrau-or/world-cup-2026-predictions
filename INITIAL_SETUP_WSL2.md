# Configuración inicial en WSL2

## 1. Paquetes base

```bash
sudo apt update
sudo apt install -y git gh curl build-essential unzip
```

Configura Git dentro de WSL2, aunque ya esté configurado en Windows:

```bash
git config --global user.name "TU NOMBRE"
git config --global user.email "TU_EMAIL_DE_GITHUB"
git config --global init.defaultBranch main
git config --global core.autocrlf input
```

## 2. Directorio de trabajo

Para mejor rendimiento, guarda el repositorio en el filesystem Linux y no bajo `/mnt/c`:

```bash
mkdir -p ~/projects
cd ~/projects
```

## 3. Instalar uv y Codex

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Cierra y abre la terminal si `uv` o `codex` todavía no aparecen en el `PATH`.

## 4. Crear el repositorio desde este starter

```bash
cp -r /RUTA/AL/STARTER ~/projects/world-cup-2026-predictions
cd ~/projects/world-cup-2026-predictions

cp .env.example .env
uv sync --group dev
uv run wc2026 doctor
uv run ruff check .
uv run mypy src
uv run pytest
```

## 5. Opcional: inicializar Git y GitHub

```bash
git init
git add .
git commit -m "chore: initialize World Cup prediction project"

gh auth login
gh repo create world-cup-2026-predictions \
  --public \
  --source=. \
  --remote=origin \
  --push
```

Puedes sustituir `--public` por `--private`.

Estos comandos son solo para publicar tu propia copia del repositorio. No son necesarios para
instalar, ejecutar `doctor` ni pasar las verificaciones locales.

## 6. Flujo de trabajo recomendado

```bash
git switch -c feat/bootstrap-cli
codex
```

Pega el prompt de Fase 0 de `prompts/CODEX_PROMPTS.md`. Después revisa:

```bash
git status
git diff
uv run ruff check .
uv run mypy src
uv run pytest
```

Cuando estés satisfecho:

```bash
git add -p
git commit -m "feat: add project bootstrap and doctor command"
git push -u origin feat/bootstrap-cli

gh pr create --fill
```
