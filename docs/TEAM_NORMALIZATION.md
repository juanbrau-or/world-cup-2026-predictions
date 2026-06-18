# Normalizacion de selecciones historicas

Fase 1C separa el catalogo canonico de equipos de los alias observados en fuentes. La fuente
auditada es `martj42/international_results` en la revision fijada en `configs/sources.yaml`.

## Estrategia

- `data/static/teams.csv` define una entidad canonica por equipo que el proyecto decide reconocer.
- `data/static/team_aliases.csv` contiene alias exactos por fuente. No usa fuzzy matching.
- Los identificadores canonicos son slugs ASCII estables y legibles derivados del nombre canonico.
- Cada nombre original observado en `home_team` o `away_team` debe tener un alias exacto para
  `international_results_csv`.
- La ingesta conserva `home_team_name_original` y `away_team_name_original`; los IDs canonicos no
  reemplazan el texto historico de la fuente.

## Campos de catalogo

- `team_status`: `current`, `historical` o `special`.
- `team_type`: distingue selecciones nacionales, territorios, estados desaparecidos,
  representantes regionales, pueblos/diasporas, entidades disputadas, clubes/comunidades y otros
  casos especiales.
- `notes`: documenta por que una entidad no debe fusionarse automaticamente.

## Decisiones historicas

- `Germany` y `German DR` son entidades separadas. No se usa el nombre de un pais actual para
  reescribir hechos historicos de otra seleccion.
- `Czechoslovakia`, `Yugoslavia`, `Saarland`, `Manchukuo`, `North Vietnam`, `Vietnam Republic`,
  `South Yemen` y `Yemen DPR` se conservan como equipos historicos propios.
- `Congo` y `DR Congo` son IDs diferentes aunque sus nombres sean similares.
- `North Korea` y `South Korea` son IDs diferentes; no existe alias generico `Korea`.
- `Vietnam`, `North Vietnam` y `Vietnam Republic` son IDs diferentes.
- `Yemen`, `South Yemen` y `Yemen DPR` son IDs diferentes.
- `Czech Republic` no se fusiona con `Czechoslovakia`.
- Equipos regionales, de pueblos, entidades disputadas y equipos no afiliados se conservan como
  `special`; no se eliminan filas ni se reemplazan por un pais moderno.
- Algunos nombres modernos aparecen en registros antiguos con identidad deportiva incierta. Esos
  casos no deben reasignarse automaticamente por continuidad politica: `Russia`, `Serbia`,
  `Ukraine` u otras selecciones solo se moveran a entidades historicas si existe evidencia explicita
  y una regla documentada.
- El dataset de modelado moderno definira una fecha minima y reglas de elegibilidad. El historico
  crudo se conserva sin alteraciones; cualquier reconciliacion futura debe vivir en una capa
  derivada y auditable.

## Auditoria

Ejecuta:

```bash
uv run wc2026 audit aliases
```

El comando valida duplicados, conflictos de alias, alias hacia IDs inexistentes, IDs canonicos sin
alias y nombres originales no resueltos. Puede escribirse un reporte JSON con `--report`.

Para la revision auditada, la resolucion de alias es 336/336 nombres originales y 49,477/49,477
filas de `results.csv`, es decir, 100.00%. Esta metrica solo mide que los nombres observados
resuelven a IDs canonicos; no mide si la fila puede entrar al dataset canonico.

La ingesta valida cubre 48,746/49,477 filas de `results.csv`, es decir, 98.52%. La diferencia se
explica por reglas de contrato ajenas a alias: `shootouts.csv` tiene 678 filas, 677 coinciden con
partidos de `results.csv` y provocan cuarentena, una no tiene partido correspondiente, existen 52
marcadores faltantes y 2 duplicados conflictivos.
