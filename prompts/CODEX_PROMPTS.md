# Biblioteca de prompts para Codex

## Cómo usar este archivo

1. Inicia Codex desde la raíz del repositorio.
2. Pídele que lea `AGENTS.md` y el prompt de una sola fase.
3. En tareas complejas, comienza en modo plan.
4. Revisa el plan antes de permitir cambios amplios.
5. Al terminar, solicita tests, revisión del diff y un resumen de riesgos.

---

## Prompt maestro de una sesión

```text
Trabaja como ingeniero de ML responsable de este repositorio. Lee primero AGENTS.md,
README.md, docs/ROADMAP.md y docs/DATA_CONTRACT.md. Inspecciona el estado del código y del git
diff. La tarea de esta sesión es: [DESCRIBIR UNA SOLA TAREA].

Antes de editar:
1. Explica qué ya existe y qué falta.
2. Identifica riesgos de data leakage, reproducibilidad y compatibilidad.
3. Propón un plan breve con archivos a modificar y pruebas a ejecutar.

Durante la implementación:
- Limítate al alcance solicitado.
- No agregues secretos, datos grandes ni artefactos generados al repositorio.
- Mantén datos crudos inmutables y transformaciones reproducibles.
- Añade o actualiza pruebas.

Al finalizar:
- Ejecuta ruff, mypy y pytest.
- Revisa el diff como si fuera un pull request.
- Resume archivos modificados, decisiones, comandos ejecutados, resultados y trabajo pendiente.
- No hagas commit ni push.
```

---

## Fase 0 — Completar el bootstrap

```text
Lee AGENTS.md y docs/ROADMAP.md. Implementa únicamente la Fase 0.

Objetivos:
- Verificar y corregir el pyproject.toml para que `uv sync --group dev` funcione.
- Implementar una CLI Typer con `wc2026 doctor`.
- `doctor` debe comprobar versión de Python, existencia de directorios esenciales,
  disponibilidad de configuración y mostrar mensajes claros sin exponer secretos.
- Crear configuración tipada con pydantic-settings.
- Crear pruebas unitarias de doctor y configuración.
- Mantener placeholders explícitos para fases futuras; no implementar ingestión ni modelos.
- Confirmar que CI, ruff, mypy y pytest funcionan.

Primero presenta el plan. Luego implementa, prueba y revisa el diff. No hagas commit.
```

## Fase 1A — Selección y contrato de fuente histórica

```text
No escribas todavía un scraper. Investiga el código existente y diseña una interfaz de fuente de
datos histórica compatible con docs/DATA_CONTRACT.md.

Entrega:
- Una decisión documentada sobre la primera fuente y una alternativa.
- Interfaz tipada para clientes y adaptadores.
- Esquemas de entrada/salida.
- Estrategia de snapshots, checksums, caché e idempotencia.
- Manejo de alias de selecciones.
- Tests con fixtures locales pequeños, sin depender de internet.

No descargues datasets grandes ni inventes endpoints. Señala qué información debe confirmar el
usuario antes de conectar una API real.
```

## Fase 1B — Ingestión histórica reproducible

```text
Implementa el adaptador histórico elegido y el pipeline para producir `matches.parquet`.

Requisitos:
- Guardar la respuesta o archivo original en `data/raw/<source>/<retrieved_at>/`.
- Crear un manifiesto con fuente, URL lógica, timestamp, checksum y versión del esquema.
- No mutar datos raw.
- Normalizar a UTC y a identificadores de equipo canónicos.
- Detectar duplicados, resultados imposibles y fechas inválidas.
- Fallar con mensajes accionables.
- Tests unitarios con datos locales y una prueba de integración opcional marcada.
- Documentar el comando exacto para reconstruir el dataset.

No agregues el dataset descargado al commit.
```

## Fase 2 — Elo sin leakage

```text
Implementa un baseline Elo prepartido.

Requisitos:
- Ordenar estrictamente por kickoff.
- Registrar rating de ambos equipos antes de actualizar con el resultado.
- Configurar rating inicial, K y pesos por competencia.
- Manejar sede neutral y ventaja real de anfitrión como conceptos distintos.
- Convertir ratings en probabilidades 1X2 mediante un método documentado.
- Agregar tests que demuestren que un partido no influye en sus propias features.
- Implementar backtest temporal y reportar log loss y Brier score.
- Incluir un baseline ingenuo para comparar.

No optimices hiperparámetros sobre el conjunto de prueba final.
```

## Fase 3 — Poisson y Dixon–Coles

```text
Implementa primero un modelo Poisson interpretable para goles y después evalúa una corrección
Dixon–Coles, sin asumir que necesariamente mejora.

Requisitos:
- Estimar fortalezas de ataque y defensa con regularización.
- Permitir ponderación temporal configurable.
- Generar matriz de marcadores truncada con control del error de masa.
- Derivar probabilidades 1X2 y goles esperados.
- Evaluar por ventanas temporales.
- Comparar contra Elo usando las mismas fechas y partidos.
- Pruebas de probabilidades válidas, determinismo y ausencia de leakage.
```

## Fase 4A — Descanso y viaje

```text
Diseña e implementa features prepartido de fatiga logística.

Features iniciales:
- horas y días desde el partido previo;
- distancia Haversine desde la sede anterior;
- viaje acumulado en 7 y 14 días;
- cambio aproximado de huso horario;
- indicador de tiempo extra en el partido anterior, solo cuando el dato exista.

Requisitos:
- No imputar datos desconocidos con valores que signifiquen “sin fatiga”.
- Añadir indicadores de missingness cuando corresponda.
- Documentar supuestos.
- Tests con itinerarios sintéticos y límites de fecha.
- Preparar una prueba de ablación, no afirmar causalidad.
```

## Fase 4B — Altitud y clima

```text
Implementa la capa de sedes y features de altitud/clima.

Requisitos:
- Tabla estática versionada de estadios con latitud, longitud, elevación, timezone y techo.
- Separar elevación de la sede, cambio respecto a la sede previa y posible aclimatación.
- No usar la altitud de la capital como proxy automático de una selección.
- Para predicciones históricas, utilizar únicamente información climática disponible en el
  instante simulado o marcar explícitamente la aproximación.
- Cachear llamadas externas.
- Incluir tests sin internet mediante mocks.
- Evaluar valor incremental mediante ablación temporal.
```

## Fase 5 — LightGBM y calibración

```text
Crea un modelo challenger LightGBM para 1X2 utilizando únicamente features prepartido ya
validadas.

Requisitos:
- Split temporal y tuning solamente dentro del periodo de entrenamiento.
- Manejo explícito de categorías y missing values.
- Calibración en un periodo posterior al entrenamiento y anterior al test.
- Comparar probabilidades sin calibrar y calibradas.
- Reportar log loss, Brier, RPS y diagramas de calibración.
- Guardar metadatos de features, cutoff, parámetros y versión.
- No reemplazar el baseline si no mejora de manera robusta.
```

## Fase 6 — Actualización con resultados 2026

```text
Implementa una actualización idempotente de fixtures y resultados del Mundial 2026.

Requisitos:
- Cliente desacoplado del proveedor.
- API key solamente desde variables de entorno.
- Timeout, retries acotados, rate limiting y caché.
- Snapshot de cada respuesta con `retrieved_at_utc`.
- No reescribir snapshots históricos.
- Detectar partidos nuevos o correcciones del proveedor.
- Actualizar datasets y predicciones sin duplicados.
- Cada predicción debe registrar `generated_at_utc`, `data_cutoff_utc` y `model_version`.
- Tests con respuestas simuladas para partido programado, en vivo, finalizado y corregido.
```

## Fase 7 — Simulación del torneo

```text
Implementa simulación Monte Carlo del torneo usando probabilidades o distribuciones de goles del
modelo, respetando el formato real configurado en archivos de datos.

Requisitos:
- Separar lógica de fase de grupos, desempates y eliminatorias.
- No hardcodear cruces dentro de funciones si pueden representarse como configuración.
- Manejar 90 minutos, tiempo extra y penales como etapas diferentes.
- Seed configurable.
- Validar que las probabilidades de campeón sumen aproximadamente 1.
- Producir tablas de probabilidad de avanzar a cada ronda.
- Tests de escenarios pequeños deterministas antes del torneo completo.
```

## Prompt de revisión antes de commit

```text
Revisa todos los cambios no confirmados como un pull request. Lee AGENTS.md.
Busca prioritariamente:
- data leakage temporal;
- uso accidental de información posterior al kickoff;
- errores de ordenamiento o timezone;
- probabilidades inválidas o mal calibradas;
- mutación de datos raw;
- secretos o datos grandes listos para commit;
- falta de tests, errores de tipado y comportamiento no idempotente.

No edites todavía. Devuelve hallazgos ordenados por severidad, con archivo y línea aproximada.
Después propón el parche mínimo para corregirlos.
```
