# Ingesta viva y snapshots del Mundial 2026

La ingesta viva no modifica datos raw ni pretende reconciliar proveedores. Una ejecución consulta
un endpoint de colección, conserva los bytes de respuesta y conserva también cada fixture miembro
con su propio manifiesto. El manifiesto contiene `provider`, `endpoint`, `source_fixture_id`,
`fetched_at`, checksum SHA-256 y `schema_revision`. El identificador `__collection__` identifica el
manifiesto de la respuesta completa; los manifiestos bajo `fixtures/` usan el identificador real de
la fuente.

Los directorios raw y las tablas canónicas por snapshot son append-only. Si se repite exactamente
el mismo snapshot, se comprueba el contenido y se reutiliza; si el contenido o el instante de fetch
cambian, se crea otro directorio. La tabla `data/processed/world_cup_2026/matches.parquet` es solo
la vista operativa más reciente. Por ello nunca sirve como evidencia de qué se sabía antes: para eso
se usa `data/processed/world_cup_2026/snapshots/<fetched_at>_<checksum>/matches.parquet` y su raw
correspondiente.

Los equipos se resuelven solo con un alias exacto vigente para
`world_cup_2026_<provider>` o con una coincidencia exacta del nombre canónico del catálogo. Un
nombre no resuelto se omite de la tabla derivada y queda en
`data/interim/world_cup_2026_unresolved_teams.json`; no existe fuzzy matching.

`data_cutoff_utc` de cada fila viva es igual a `fetched_at`. Es una propiedad de la observación, no
una feature ni una autorización para usar un resultado que fue conocido después del kickoff.

La fuente secundaria, cuando está configurada, solo genera un reporte de discrepancias de equipos,
kickoff y marcador. No genera una segunda tabla operativa ni combina datos de proveedores.

## Participantes por determinar

Un fixture futuro con `homeTeam.name` y `awayTeam.name` explícitamente nulos se conserva como
`participants_status: "tbd"` cuando su estado es programado. No se intenta resolver aliases ni se
crean IDs canónicos sintéticos para esos cruces. La vista se publica en
`data/interim/world_cup_2026_tbd_fixtures.json`; incluye el id de la fuente, kickoff, estado,
stage, actualización de origen, marcador nulo y la ruta al snapshot raw individual. Estos fixtures
no entran en `matches.parquet` y por tanto no son candidatos a predicción.

Un fixture con ambos nombres no nulos tiene `participants_status: "known"` y sigue la resolución
exacta normal. Un nombre presente y el otro nulo tiene `participants_status: "partially_known"`:
se conserva sin resolver aliases para el lado nulo, no entra en `matches.parquet` y se informa en
`data/interim/world_cup_2026_partially_known_fixtures.json`. Nombres malformados, participantes
ausentes en partidos activos o finales, o un contrato de marcador inválido son registros inválidos;
se informan en `data/interim/world_cup_2026_invalid_fixtures.json` sin interrumpir los demás
registros.

Cada colección se procesa por fixture. Los snapshots raw y las tablas bajo `snapshots/` permanecen
append-only. Cuando el mismo `source_fixture_id` pasa de TBD a conocido, se conserva el snapshot
anterior, se crea otro y la vista operativa se sobrescribe con como máximo una versión vigente por
id de origen.
