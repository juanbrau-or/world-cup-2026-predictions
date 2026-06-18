# Contrato de datos

Este documento define la representacion canonica de partidos internacionales de selecciones
nacionales para la Fase 1A. El contrato almacena hechos observados y metadatos de trazabilidad; no
incluye features predictivas, ratings, variables de viaje, clima, altitud ni salidas de modelos.

La implementacion tipada vive en `src/worldcup2026/data/contracts.py` y la interfaz de fuentes
historicas vive en `src/worldcup2026/data/sources.py`.

## Principios

- Un registro canonico representa un partido entre dos selecciones nacionales.
- El equipo listado como local es el `designated_home` de la fuente o competicion, no una senal
  automatica de ventaja de localia real.
- Los nombres originales de la fuente se conservan junto a identificadores canonicos de equipos.
- La fecha, hora y zona horaria deben expresar la precision real disponible en la fuente.
- Los goles a 90 minutos, goles tras tiempo extra y penales se guardan en campos separados.
- Los datos crudos no se corrigen en sitio; las normalizaciones pertenecen a capas derivadas.
- Las tablas derivadas deben ser reproducibles e idempotentes a partir de fuente, identificador de
  fuente y version de esquema.

## Tabla canonica de partidos

Version actual del esquema: `international_match_v1`.

| Campo | Tipo | Requerido | Regla |
|---|---:|---:|---|
| `match_id` | string | si | Identificador estable dentro del proyecto. No debe duplicarse en una coleccion. |
| `schema_version` | string | si | Debe ser `international_match_v1`. |
| `match_status` | enum | si | `played`, `scheduled`, `postponed`, `cancelled`, `suspended`, `abandoned`. |
| `match_date` | date | si | Fecha nominal/local del partido segun la fuente o competicion. Puede diferir de la fecha UTC. |
| `kickoff_utc` | datetime UTC nullable | segun precision | Obligatorio si `kickoff_time_status=exact_utc`; nulo en registros de solo fecha u hora local sin zona. |
| `kickoff_local_time` | string `HH:MM` nullable | no | Hora local observada cuando no se puede convertir de forma segura a UTC. |
| `kickoff_timezone` | string nullable | no | Zona horaria IANA u offset reportado por la fuente cuando exista. |
| `kickoff_time_status` | enum | si | `exact_utc`, `date_only`, `local_time_without_timezone`. |
| `home_team_name_original` | string | si | Nombre local/designado tal como aparece en la fuente. |
| `away_team_name_original` | string | si | Nombre visitante/designado tal como aparece en la fuente. |
| `home_team_id` | string | si | Identificador canonico del equipo local/designado. |
| `away_team_id` | string | si | Identificador canonico del equipo visitante/designado. |
| `home_goals_90` | integer nullable | si para jugados | Goles del local/designado al final del tiempo reglamentario. |
| `away_goals_90` | integer nullable | si para jugados | Goles del visitante/designado al final del tiempo reglamentario. |
| `result_90` | enum nullable | si para jugados | `home_win`, `draw`, `away_win`; debe coincidir con los goles a 90. |
| `extra_time_played` | boolean | si | `true` solo si hubo tiempo extra. |
| `home_goals_after_extra_time` | integer nullable | si si hubo TE | Goles del local/designado despues del tiempo extra. |
| `away_goals_after_extra_time` | integer nullable | si si hubo TE | Goles del visitante/designado despues del tiempo extra. |
| `penalty_shootout` | boolean | si | `true` solo si hubo definicion por penales. |
| `home_penalty_goals` | integer nullable | si si hubo penales | Penales anotados por el local/designado. |
| `away_penalty_goals` | integer nullable | si si hubo penales | Penales anotados por el visitante/designado. |
| `competition` | string | si | Torneo o competicion reportada. |
| `stage` | string nullable | no | Fase o ronda cuando la fuente la proporcione. |
| `match_type` | enum | si | `friendly`, `qualifier`, `continental_tournament`, `world_cup`, `other`. |
| `city` | string nullable | no | Ciudad sede cuando este disponible. |
| `host_country` | string nullable | no | Pais donde se disputa el partido. |
| `venue_name_original` | string nullable | no | Estadio o sede como aparece en la fuente. |
| `neutral_site` | boolean nullable | no | Indicador de sede neutral reportado o inferido por reglas documentadas. |
| `home_advantage_status` | enum | si | `home_team`, `away_team`, `neutral`, `shared_host`, `unknown`. |
| `source` | string | si | Nombre estable de la fuente original. |
| `source_match_id` | string | si | Identificador estable de la fuente. No debe repetirse dentro de la misma fuente. |
| `retrieved_at_utc` | datetime UTC | si | Momento de adquisicion del dato, en UTC. |

## Fuente historica inicial

Decision de Fase 1A: la primera integracion debe ser un archivo CSV estatico de resultados
internacionales historicos, identificado internamente como `international_results_csv`. La razon es
que permite reconstruccion determinista, snapshots completos, checksums simples y fixtures locales
pequenos antes de conectar una API. La fuente debe tener, como minimo, fecha, equipos designados,
marcador de tiempo reglamentario o marcador final claramente documentado, torneo, sede y neutralidad
cuando exista.

Alternativa: una API historica de partidos internacionales, identificada como
`historical_matches_api`, si el usuario confirma licencia, estabilidad de identificadores y limites
de uso. No se conectara ninguna API en Fase 1A.

Antes de implementar Fase 1B el usuario debe confirmar:

- URL o ruta logica exacta de la fuente.
- Licencia y permiso de uso para snapshots en `data/raw/`.
- Significado exacto de los campos de marcador: 90 minutos, tiempo extra y penales.
- Si el campo local/visitante representa equipo designado o localia real.
- Si la fuente provee zonas horarias, solo fechas o horas locales sin zona.
- Si los identificadores de la fuente son estables entre descargas.

La interfaz tipada de Fase 1A define:

- `HistoricalFetchRequest`: fuente, URI logica y directorio de cache.
- `RawSnapshotManifest`: fuente, URI logica, `retrieved_at_utc`, checksum SHA-256, cache key y ruta
  raw.
- `RawSnapshot`: bytes originales mas manifiesto.
- `HistoricalSourceRecord`: registro parseado previo a normalizacion canonica.
- `HistoricalDataClient` y `HistoricalMatchAdapter`: protocolos para obtener snapshots y adaptar
  registros a `CanonicalMatch`.

La idempotencia de snapshots se basa en `(source, logical_uri, content_sha256, schema_version)`.
Fase 1B debe escribir respuestas originales en `data/raw/<source>/<retrieved_at>/` con manifiesto y
nunca mutar esos bytes. La cache puede reutilizar un snapshot si el checksum y la URI logica
coinciden.

## Reglas temporales

`kickoff_time_status` controla como interpretar fecha y hora:

- `exact_utc`: `kickoff_utc` es obligatorio, timezone-aware y UTC. `match_date` conserva la fecha
  nominal/local de la fuente o competicion, por lo que puede diferir de la fecha UTC en partidos
  nocturnos. Puede conservarse `kickoff_timezone` si la fuente informo la zona usada para convertir,
  pero la clave temporal segura es `kickoff_utc`. `kickoff_timezone` debe ser una zona IANA valida
  o un offset UTC explicito.
- `date_only`: solo se conoce la fecha. `kickoff_utc`, `kickoff_local_time` y
  `kickoff_timezone` deben ser nulos.
- `local_time_without_timezone`: se conoce una hora local `HH:MM`, pero falta una zona horaria
  confiable. `kickoff_utc` y `kickoff_timezone` deben ser nulos.

Para features y backtesting futuros, las tablas de entrenamiento deberan usar `match_id`,
`kickoff_utc` y `data_cutoff_utc`. Los registros sin `kickoff_utc` no son seguros para features
prepartido hasta que una capa posterior resuelva la hora de forma documentada.

La validacion de colecciones puede exigir orden temporal por `kickoff_utc` cuando ambos registros
tienen hora exacta, usando solo `match_date` cuando falta precision horaria. No debe inventarse
medianoche UTC para ordenar registros de solo fecha o de hora local sin zona. No debe usarse un split
aleatorio de partidos.

## Reglas de marcadores

- Un partido `played` requiere `home_goals_90`, `away_goals_90` y `result_90`.
- `result_90` debe coincidir con los goles a 90 minutos.
- Partidos `scheduled`, `postponed`, `cancelled`, `suspended` y `abandoned` no deben incluir
  marcadores, resultado a 90, tiempo extra ni penales.
- Si `extra_time_played=false`, los campos `*_goals_after_extra_time` deben ser nulos.
- Si `extra_time_played=true`, el marcador a 90 debe estar empatado y ambos campos de goles tras
  tiempo extra son obligatorios y no pueden ser menores que los goles a 90.
- Si `penalty_shootout=false`, los campos `*_penalty_goals` deben ser nulos.
- Si `penalty_shootout=true`, ambos campos de penales son obligatorios y deben identificar un
  ganador.
- Si hubo penales despues de tiempo extra, el marcador tras tiempo extra debe estar empatado.
- Si hubo penales sin tiempo extra, el marcador a 90 debe estar empatado.

## Localia y neutralidad

`home_team_id` y `away_team_id` identifican al equipo local/visitante designado por la fuente o por
la competicion. No implican ventaja de localia.

La localia real se expresa con:

- `neutral_site`: indicador booleano de sede neutral cuando se conoce; puede ser nulo si la fuente no
  lo permite determinar.
- `home_advantage_status`: enum que distingue si la ventaja corresponde al equipo listado como local,
  al visitante, a una sede neutral, a anfitriones compartidos o si es desconocida.

Si `neutral_site=true`, `home_advantage_status` debe ser `neutral` o `shared_host`. Si
`neutral_site=false`, `home_advantage_status` no puede ser `neutral`. Si la fuente no permite saber
si la sede fue neutral, `neutral_site` debe ser nulo.

## Identificadores y alias

Los identificadores canonicos de equipos deben ser estables y separarse de los nombres originales.
La tabla inicial de demostracion es `data/static/team_aliases.csv`:

| Campo | Descripcion |
|---|---|
| `canonical_team_id` | Identificador canonico usado en contratos derivados. |
| `canonical_name` | Nombre canonico vigente o preferido. |
| `source` | Fuente o contexto donde aparece el alias. |
| `source_name` | Nombre exacto observado en esa fuente. |
| `valid_from` | Fecha desde la cual aplica el alias, si se conoce. |
| `valid_to` | Fecha final de vigencia, si se conoce. |

Esta tabla no intenta resolver todos los paises en Fase 1A. Su objetivo es demostrar como preservar
nombres historicos o alternativos como `West Germany` sin perder el identificador canonico usado por
el proyecto.

Los identificadores canonicos deben usar el patron `^[a-z][a-z0-9_]*$`. Cuando se proporcione una
tabla de alias a la validacion de colecciones, cada par `(source, source_name, match_date)` debe
resolver a un unico `canonical_team_id` vigente por `valid_from` y `valid_to`.

## Validacion de colecciones

Una coleccion valida debe cumplir:

- Cada registro individual valida contra `CanonicalMatch`.
- `match_id` no se repite.
- El par `(source, source_match_id)` no se repite.
- Opcionalmente, la coleccion puede exigirse en orden temporal no decreciente.

Registros duplicados provenientes de varias fuentes no se fusionan en esta fase. Una etapa posterior
de reconciliacion debera decidir si dos `source_match_id` distintos representan el mismo partido y
asignar o reutilizar un `match_id` estable.

## Compatibilidad con Parquet

El contrato usa escalares simples: strings, enteros no negativos, booleanos, fechas y datetimes UTC.
Los enums se serializan como strings. Los valores faltantes se representan como nulos; no se deben
usar sentinelas como `-1`, `"unknown"` en campos de texto nullable ni fechas artificiales.

## Anti-leakage

Este contrato puede almacenar resultados observados porque describe partidos historicos, pero esos
campos no deben entrar a features prepartido. En fases posteriores:

- Las features deben calcularse con datos disponibles antes de `kickoff_utc`.
- Toda tabla de entrenamiento debe incluir `match_id`, `kickoff_utc` y `data_cutoff_utc`.
- Los rolling features deben usar solamente partidos anteriores.
- `retrieved_at_utc` registra adquisicion de datos; no reemplaza a `data_cutoff_utc` para
  predicciones.

## Contratos futuros no implementados en Fase 1A

### Tabla de features prepartido

Debe conservar, como minimo:

- `match_id`
- `kickoff_utc`
- `data_cutoff_utc`
- version del codigo o feature set
- variables calculadas antes del kickoff

### Tabla de predicciones

Campos minimos previstos:

- `generated_at_utc`
- `data_cutoff_utc`
- `match_id`
- `model_name`
- `model_version`
- `prob_home_win`
- `prob_draw`
- `prob_away_win`
- `expected_home_goals` cuando aplique
- `expected_away_goals` cuando aplique

Las tres probabilidades deben ser finitas, estar en `[0, 1]` y sumar 1 dentro de una tolerancia.
