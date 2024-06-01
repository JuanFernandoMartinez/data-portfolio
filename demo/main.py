import tkinter as tk
import os
from tkinter import filedialog  # Importar módulo para selección de archivos
import subprocess
import pickle
import pandas as pd


ruta_imagen = ""
# Función para cargar una imagen
def cargar_imagen():
    # Abrir ventana de selección de archivos
    ruta_imagen = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=(("Archivos de imagen", "*.png *.jpg *.jpeg"), ("Todos los archivos", "*.*"))
    )

    # Verificar si se seleccionó una imagen
    if ruta_imagen:
        try:
            # Cargar la imagen
            imagen = tk.PhotoImage(file=ruta_imagen)
            
            # Actualizar la imagen en la etiqueta
            etiqueta_imagen.config(image=imagen)
            etiqueta_imagen.image = imagen  # Mantener referencia a la imagen

            # Deshabilitar el botón de carga si ya se ha cargado una imagen
            
            boton_procesar.config(state="normal")

        except Exception as error:
            # Mostrar mensaje de error en caso de problemas
            tk.messagebox.showerror("Error", f"Error al cargar la imagen: {error}")

# Función para procesar la imagen (funcionalidad futura)
def procesar_imagen():

    #step 1 save image to 
    path_to_image = os.path.join("cancerSeno_bw\\test\\cancer", "test.png")
    imagen = etiqueta_imagen.image
    imagen.write(path_to_image)
    
    proceso1 = subprocess.Popen(["python", "testFeature.py", path_to_image], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proceso1.wait()
    proceso2 = subprocess.Popen(["python", "testDataSetConstruction.py", path_to_image], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proceso2.wait()

    #step 3 load the dataframe 
    df = pd.read_csv(os.path.join("featuresTestFromTest", "combined_features_test.csv"))
    

    #step 4 load the model
    with open('GBoost_tunned.pkl','rb') as file:
        model = pickle.load(file)

    print(type(model))
    #step 5 predict
    X = df[['pixel_density', 'brightness', 'std_deviation']]
    prediction = model.predict(X)

    if prediction[0] == 0:
        tk.messagebox.showinfo("Resultado", "No se detectó cáncer de seno")
    else:
        tk.messagebox.showinfo("Resultado", "Se detectó cáncer de seno")
    print(prediction[0])
    #run the external scripts
      # Reemplazar con el código de procesamiento de imagen

# Crea la ventana principal
ventana = tk.Tk()
ventana.geometry("200x200")
ventana.title("Breast Cancer Detection System")

# Crea una etiqueta para mostrar la imagen
etiqueta_imagen = tk.Label(ventana)


etiqueta_imagen.pack()

# Crea un botón para cargar la imagen
boton_cargar = tk.Button(ventana, text="Cargar imagen", command=cargar_imagen)
boton_cargar.pack()

# Crea un botón para procesar la imagen (funcionalidad futura)
boton_procesar = tk.Button(ventana, text="Procesar imagen", command=procesar_imagen, state="disabled")
boton_procesar.pack()

# Inicia el bucle principal de la interfaz
ventana.mainloop()
