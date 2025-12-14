
# Sistema de Recuperación de Imágenes Basado en Contenido (CBIR)

## Autores

- **Ignacio Díaz Hernanz**
- **Néstor Rovira Geijo**   

---

## Introducción

Este repositorio contiene el código y los recursos desarrollados para un proyecto de Recuperación de Imágenes Basado en Contenido (CBIR). 

El objetivo del proyecto es diseñar e implementar un sistema capaz de recuperar imágenes similares a una consulta en función de su contenido visual, utilizando técnicas de extracción de características y búsqueda por similitud.

El repositorio incluye el flujo completo del sistema, desde la preparación de los datos y la extracción de características hasta la indexación y recuperación de imágenes, junto con una interfaz web interactiva desarrollada con Streamlit.

---

## Instalación y configuración

### 1. Clonar el repositorio

```
git clone https://github.com/NestorRovira/AAPI_CBIR.git
cd AAPI_CBIR 
```

---

### 2. Preparación del dataset

Dentro de la carpeta `data/` se proporciona un archivo comprimido llamado `raw.zip`.

Es necesario:
- Descomprimir el archivo `raw.zip` **dentro de la carpeta `data/`**.
- Tras descomprimir, debe existir una carpeta llamada `raw` que contenga las imágenes organizadas las carpetas (`train`) y (`test`).


---

### 3. Configuración del entorno (Conda)

El proyecto utiliza **Conda** para la gestión del entorno y las dependencias.

1. Crear el entorno a partir del archivo de configuración proporcionado:

```
conda env create -f environment.yml  
```

2. Activar el entorno:

```
conda activate cbir_env  
```

---

### 4. Ejecución de la aplicación

Una vez configurado el entorno y preparado el dataset, la aplicación se puede lanzar con el siguiente comando:

```
python -m streamlit run app.py  
```

Desde la interfaz web es posible:
- Seleccionar distintos extractores de características.
- Comparar su comportamiento y resultados.
- Realizar consultas utilizando imágenes del conjunto de prueba (`test`) para evaluar visualmente la calidad de la recuperación.


---
