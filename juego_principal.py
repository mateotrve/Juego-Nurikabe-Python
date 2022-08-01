#Programa creado por: Mateo Andres Arenas Angel, Juan David Ardila Paniagua
#Pontificia Universidad Javeriana
#Juego Nurikabe

from cgitb import text
from distutils.dir_util import copy_tree
from functools import partial
from tkinter import *
from tkinter import messagebox
import numpy as np
from juego_sintetico import *

columnas = 0
filas = 0
botones = []
numeros_tablero = 0

#Funcion Juego Sintetico
def juegoSintetico(lineas):
    sintetico(lineas)

#Funcion click en casilla
def enventoCasilla(n_button):
    if(n_button["background"] == "white"):
        n_button.configure(background="black", fg="white")
    else:
        n_button.configure(background="white", fg="black")

def eventoCalcular():
    #Primera regla: Una region esta llena si ya tiene su limite de celdas, para una celda blanca, el limite lo dice el numero.
    #Para una celda negra el limite es el numero de celdas del tablero menos la suma de los numeros escritos.
    global columnas
    global filas
    total = filas*columnas
    
    limiteNegras = total - numeros_tablero
    cont1 = 0
    
    for i in range(0,filas):
        for j in range(0,columnas):
            if(botones[i][j]["background"] == "black"):
                cont1 = cont1+1
    
    if(limiteNegras==cont1):
        messagebox.showinfo("Ganaste","El juego esta resuelto") 
    else:
        messagebox.showinfo("Incompleto","El juego no esta resuelto todavia, sigue jugando")
    
    cont1 = 0
    #Segunda regla: Una región blanca sin número tiene hambre. (Pero su límite es desconocido).
    #Si hay tres celdas negras en forma de "L", la cuarta celda del cuadrado debe ser blanca. 
    #Si hay un cuadrado con dos celdas negras y dos celdas sin resolver, 
    #verifique si una de las celdas sin resolver puede ser blanca. Si no, la otra celda sin resolver debe ser blanca.
    subMatriz = []
    cont2 = 0
    for k in range(2):
        subMatriz.append([0]*2)
        
    for i in range(0,filas-1):
        for j in range(0,columnas-1):
            subMatriz[0][cont1] = botones[i][j]
            cont1 = cont1+1
            if(cont1 == 2):
                subMatriz[1][0] = botones[i+1][j-1]
                subMatriz[1][1] = botones[i+1][j]
                for q in range(2):
                    for t in range(2):
                        if(botones[i][j]["background"] == "black"):
                            cont2 = cont2 + 1
                        if(cont2 == 4):
                            messagebox.showinfo("Hay piscinas","Debe haber un solo mar el cual no puede contener piscinas, sigue jugando")
                    cont2 = 0
                cont1 = 0
                                       
            
        

#Funcion crear tablero
def abrirTablero(nombre_a, ventana_principal, opcion):
    ventana_principal.destroy()

    try:
        archivo=open(nombre_a, "r")
        lineas = archivo.readlines()
    except IOError:
        #Mensaje de error al no encontrar el archivo que escribio el usuario
        messagebox.showwarning("Error", "El archivo no se ha encontrado, compruebe el nombre del archivo. (El nombre debe incluir la extension .txt)")
        iniciarJuego()

    point= lineas[0].strip().split(",")
    
    if(opcion == 2):
        #Los objetos messagebox son ventanas de informacion o error, esta es cuando el usuario elige modo sintetico
        juegoSintetico(lineas)
    else:
        global columnas
        columnas = int(point[1])
        global filas
        filas = int(point[0])
        
        #Tablero es la raiz de la ventan del juego
        tablero = Tk()
        tablero.title("Nurikabe")
        #Matriz es solo la cuadricula del juego
        matriz = Frame(tablero)
        matriz.pack()
        global botones
        global numeros_tablero
        
        for t in range(filas):
            botones.append([0]*columnas)
        
        
        for r in range(0, filas):
            for c in range(0, columnas):  
                #El siguiente ciclo es para que revise el archivo cada vez que vaya creando un boton para saber si toca colocarle numero    
                for k in range(1, len(lineas)):
                    linea= lineas[k].strip().split(",")
                    if(int(linea[1]) == r+1 and int(linea[2]) == c+1):
                        n_button = Button(matriz, width=7, height=3, background="white" ,foreground="black", text=linea[0])
                        n_button.grid(row=r, column=c)
                        numeros_tablero = numeros_tablero+int(linea[0])
                        break
                    else:
                        n_button = Button(matriz, width=7, height=3, background="white" ,foreground="red")
                        n_button.grid(row=r, column=c)
                        n_button["command"] = partial(enventoCasilla, n_button)
                botones[r][c] = n_button
        Label(tablero).pack()

        #Boton que llamara la funcion de verificar
        calcular = Button(tablero, width=13, height=3, bg="black", fg="white", text="Verificar", font=("Verdana",12))
        calcular["command"] = partial(eventoCalcular)
        calcular.pack()

        tablero.mainloop()

#Funcion para iniciar la aplicación, menu de inicio
def iniciarJuego():
    #ventanaPrincipal es la raiz de la venta menu inicial
    ventana_principal = Tk()
    ventana_principal.config(width=300, height=300)
    ventana_principal.title("Nurikabe Game")

    titulo = Label(ventana_principal, text="NURIKABE", bg="Black", fg="white", font=("Verdana",36))
    titulo.pack(fill=X)
    #Lo de la linea de abajo es para separa los componentes de la ventana y no se vea tan pegado
    Label(ventana_principal).pack()

    #En la variable archivo se guarda lo que escribio el usuario en la caja de texto
    archivo = StringVar()
    frame1 = Frame(ventana_principal)
    frame1.pack()
    #campText es la caja de texto donde el usuario escribe el nombre del archivo
    camp_text = Entry(frame1, font=("Verdana",12), textvariable=archivo)
    camp_text.pack(side=RIGHT)
    text1 = Label(frame1, text="Escriba el nombre del archivo del tablero:", font=("Verdana",12))
    text1.pack(side=LEFT)
    Label(ventana_principal).pack()

    text2 = Label(ventana_principal, text="Seleccione el tipo de jugador:", font=("Verdana",12))
    text2.pack()
    #En la variable opcion de guarda la opcion que eligio del modo de juego, 1 para humano, 2 para sintetico
    opcion = IntVar()
    Radiobutton(ventana_principal, text="Humano", font=("Verdana",12), variable=opcion, value=1).pack()
    Radiobutton(ventana_principal, text="Sintetico", font=("Verdana",12), variable=opcion, value=2).pack()
    Label(ventana_principal).pack()

    #El boton de inicio lo que hace es llamar a otra ventan y cierra el menu inicial
    botonInicio = Button(ventana_principal, bg="black", fg="white", text="Iniciar Juego", font=("Verdana",12), command= lambda: abrirTablero(archivo.get(), ventana_principal, opcion.get()))
    botonInicio.pack()

    ventana_principal.mainloop()

iniciarJuego()