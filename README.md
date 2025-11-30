# Algoritmo Genético para Generación de Horarios Universitarios

Este proyecto implementa un **Algoritmo Genético (AG)** para generar horarios universitarios factibles y con buena calidad, dentro del contexto de la materia **Inteligencia Computacional y Aplicaciones**.

El trabajo se basa en el artículo:

> Jiang, F., Shuai, J., Nong, Y., & Gao, Y. (2025).  
> _Application of Genetic Algorithms: a Case Study on Course Scheduling_.  
> Proceedings of the 5th International Conference on Internet, Education and Information Technology (IEIT 2025).

A partir de este enfoque se implementa una versión **base** del AG y una versión **mejorada**, y se comparan sus resultados.

---

## Descripción general

- Se modela un horario semanal de **5 días (Lun–Vie)** con **6 bloques por día**.
- Se trabaja con un escenario de prueba reducido de **4 materias**, cada una con:
  - Profesor asignado
  - Aula asignada
  - Número de sesiones por semana

Cada individuo del Algoritmo Genético representa un **horario completo**, donde:

- Las **filas** son días de la semana.
- Las **columnas** son bloques de tiempo.
- Cada curso se representa como una matriz `5 x 6` con `1` si hay clase en ese bloque y `0` si no.

La **función de fitness** combina:

1. **Factibilidad**:
   - No debe haber choques de profesor.
   - No debe haber choques de aula.
2. **Calidad del horario**:
   - Cobertura de días (que las materias no queden todas en un solo día).
   - Penalización de **huecos** (en la versión mejorada).

---

## Versiones del algoritmo

Se comparan dos configuraciones:

### 1. Algoritmo Base

- Tamaño de población: `50`
- Generaciones: `500`
- Probabilidad de cruza: `0.8`
- Probabilidad de mutación: `0.05`
- Selección: **ruleta** (fitness-proporcional)
- Elitismo: se conserva el mejor individuo de cada generación
- **Sin** penalización explícita de huecos
- **Sin** mutación adaptativa

### 2. Algoritmo Mejorado

- Tamaño de población: `80`
- Generaciones: `500`
- Probabilidad de cruza: `0.8`
- Probabilidad de mutación inicial: `0.05`
- Selección: **ruleta**
- Elitismo: igual que base
- **Con** penalización de huecos en la función de fitness
- **Con** mutación adaptativa:
  - Si el algoritmo se estanca (no mejora en varias generaciones), aumenta ligeramente la tasa de mutación (hasta un máximo).

El objetivo de la versión mejorada es obtener **horarios igual de factibles**, pero con **menos huecos** y mejor organización semanal.

---

## Archivos del proyecto

- `ga_horarios.py`  
  Script principal con:

  - Definición del escenario de prueba (cursos, profesores, aulas).
  - Implementación del Algoritmo Genético.
  - Ejecución de la **versión base** y la **versión mejorada**.
  - Impresión del mejor horario encontrado en cada caso.

- `README.md`  
  Este archivo de documentación.

---

## Requisitos

- **Python** 3.9 o superior
- Biblioteca:
  - `numpy`

Instalación de dependencias (si no tienes numpy):

```bash
pip install numpy
```
