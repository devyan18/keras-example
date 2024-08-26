# Instalación de dependencias

Para instalar las dependencias necesarias para ejecutar el código de este repositorio, se puede utilizar el archivo `requirements.txt` o instalar las dependencias manualmente.


## Instalación de dependencias con `requirements.txt`

paso 1: Crear un entorno virtual
```cmd
python -m venv venv
```

paso 2: Activar el entorno virtual
```cmd
./venv/Scripts/activate
```
or
``` cmd
source venv/Scripts/activate
```

paso 3: Instalar las dependencias
```cmd
pip install -r requirements.txt
```

## Instalación de dependencias manualmente

```cmd
pip install tensorflow keras numpy matplotlib pandas
```

# Ejecución de los scripts

Para ejecutar los scripts de este repositorio, se puede utilizar el siguiente comando:

```cmd
python app.py
```