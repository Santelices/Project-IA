# Librerias
import cv2
from ultralytics import YOLO
import math

class ShopIA:
    # Init
    def init(self):
        # VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1480)
        self.cap.set(4, 920)

        # Modelos:
        # Modelo de Objetos
        modeloObjetos = YOLO('Modelos/yolov8l.onnx')
        self.modeloObjetos = modeloObjetos

        # Modelo de Billetes
        modeloBilletes = YOLO('Modelos/Dolares.onnx')
        self.modeloBilletes = modeloBilletes

        clasesObjetos = modeloObjetos.names
        # Clases de Objetos
        """clasesObjetos = ['bottle', 'spoon', 'banana', 'apple', 'orange', 'broccoli', 'carrot',
                         'cake', 'mouse', 'remote', 'cell phone', 'book', 'clock',
                         'scissors', 'toothbrush']"""
        self.clasesObjetos = clasesObjetos

        # Clases de Billetes
        clasesBilletes = ['5Dollar', '10Dollar', '20Dollar']
        self.clasesBilletes = clasesBilletes

        # Balance total
        balance_total = 0
        self.balance_total = balance_total
        self.pago = ''

        return self.cap

    # Area
    def dibujar_area(self, img, color, xi, yi, xf, yf):
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 2, 1)
        return img

    # Texto
    def dibujar_texto(self, img, color, texto, xi, yi, tamano, grosor, fondo=False):
        tamano_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_COMPLEX, tamano, grosor)
        dim = tamano_texto[0]
        linea_base = tamano_texto[1]
        if fondo:
            img = cv2.rectangle(img, (xi, yi - dim[1] - linea_base), (xi + dim[0], yi + linea_base - 7), (0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, texto, (xi, yi - 5), cv2.FONT_HERSHEY_COMPLEX, tamano, color, grosor)
        return img

    # Linea
    def dibujar_linea(self, img, color, xi, yi, xf, yf):
        img = cv2.line(img, (xi, yi), (xf, yf), color, 2, 1)
        return img

    def calcular_area(self, frame, xi, yi, xf, yf):
        # Información del fotograma
        alto, ancho, _ = frame.shape
        # Coordenadas
        xi, yi = int(xi * ancho), int(yi * alto)
        xf, yf = int(xf * ancho), int(yf * alto)

        return xi, yi, xf, yf

    #lista Mercado
    def lista_mercado(self, frame, objeto):
        lista_productos = {'Mobile phone': 50, 'Scissors': 1,'Computer mouse':20}

        # Configuración de Texto
        area_lista_xi, area_lista_yi, area_lista_xf, area_lista_yf = self.calcular_area(frame, 0.7739, 0.6250, 0.9649,0.9444)
        tamano_objeto, grosor_objeto = 0.60, 1

        # Añadir a la lista de compras con precio
        # Telofono Celular
        if objeto == 'Mobile phone' not in [item[0] for item in self.lista_compras]:
            precio = lista_productos['Mobile phone']
            self.lista_compras.append([objeto, precio])
            # Mostrar
            texto = f'{objeto} = ${precio}'
            frame = self.dibujar_texto(frame, (0, 255, 0), texto, area_lista_xi + 10,
                                       area_lista_yi + (40 + (self.posicion_productos * 20)),
                                       tamano_objeto, grosor_objeto, fondo=False)
            self.posicion_productos += 1
            # Precio
            self.precio_acumulado += precio

        if objeto == 'Scissors' not in [item[0] for item in self.lista_compras]:
            precio = lista_productos['Scissors']
            self.lista_compras.append([objeto, precio])
            # Mostrar
            texto = f'{objeto} = ${precio}'
            frame = self.dibujar_texto(frame, (0, 255, 0), texto, area_lista_xi + 10,
                                       area_lista_yi + (40 + (self.posicion_productos * 20)),
                                       tamano_objeto, grosor_objeto, fondo=False)
            self.posicion_productos += 1
            # Precio
            self.precio_acumulado += precio

        if objeto == 'Computer mouse' not in [item[0] for item in self.lista_compras]:
            precio = lista_productos['Computer mouse']
            self.lista_compras.append([objeto, precio])
            # Mostrar
            texto = f'{objeto} = ${precio}'
            frame = self.dibujar_texto(frame, (0, 255, 0), texto, area_lista_xi + 10,
                                       area_lista_yi + (40 + (self.posicion_productos * 20)),
                                       tamano_objeto, grosor_objeto, fondo=False)
            self.posicion_productos += 1
            # Precio
            self.precio_acumulado += precio

        return frame

    # Balance process
    def proceso_balance(self, tipo_billete):
        if tipo_billete == '5Dollar':
            self.balance = 5
        elif tipo_billete == '10Dollar':
            self.balance = 10
        elif tipo_billete == '20Dollar':
            self.balance = 20

    # Proceso de Pago
    def proceso_pago(self, precio_acumulado, balance_acumulado):
        pago = balance_acumulado - precio_acumulado
        if pago < 0:
            texto = f'Falta cancelar: {abs(pago):.2f}$'
        elif pago > 0:
            texto = f'Su cambio es de: {abs(pago):.2f}$'
            self.precio_acumulado = 0
            self.balance_total = 0
        elif pago == 0:
            texto = f'Gracias por su compra!'
            self.precio_acumulado = 0
            self.balance_total = 0

        return texto

    # INFERENCIA
    def modelo_prediccion(self, clean_frame, frame, modelo, clase):
        bbox = []
        cls = 0
        conf = 0
        # Yolo | AntiSpoof
        resultados = modelo(clean_frame, stream=True, verbose=False)
        for res in resultados:
            # Box
            cajas = res.boxes
            for caja in cajas:
                # Extraer coordenadas
                # Bounding box
                x1, y1, x2, y2 = caja.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Error < 0
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0

                bbox = [x1, y1, x2, y2]

                # Clase
                cls = int(caja.cls[0])

                # Confianza
                conf = math.ceil(caja.conf[0])

                if clase == 0:
                    if cls < len(self.clasesObjetos):
                        # Dibujar
                        objeto = self.clasesObjetos[cls]
                        texto_objeto = f'{self.clasesObjetos[cls]}'#{int(conf * 100)}%
                        # Lista del Mercado
                        frame = self.lista_mercado(frame, objeto)

                        # Dibujar
                        tamano_objeto, grosor_objeto = 0.75, 1
                        frame = self.dibujar_texto(frame, (0, 255, 0), texto_objeto, x1, y1, tamano_objeto,
                                                   grosor_objeto, fondo=True)
                        frame = self.dibujar_area(frame, (0, 255, 0), x1, y1, x2, y2)

                if clase == 1:
                    if cls < len(self.clasesBilletes):
                        # Dibujar
                        tipo_billete = self.clasesBilletes[cls]
                        texto_objeto = f'{self.clasesBilletes[cls]}'#{int(conf * 100)}%
                        self.proceso_balance(tipo_billete)

                        # Dibujar
                        tamano_objeto, grosor_objeto = 0.75, 1
                        frame = self.dibujar_texto(frame, (63, 190, 234), texto_objeto, x1, y1, tamano_objeto,grosor_objeto, fondo=True)
                        frame = self.dibujar_area(frame, (63, 190, 234), x1, y1, x2, y2)
        return frame

    # Main
    def tiendaIA(self, cap):
        while True:
            # Captura los frames
            ret, frame = cap.read()
            # Lee el teclado
            t = cv2.waitKey(5)

            # Frame para detección de objetos
            clean_frame = frame.copy()

            # Información de la lista del mercado
            lista_compras = []
            self.lista_compras = lista_compras
            posicion_productos = 1
            self.posicion_productos = posicion_productos
            precio_acumulado = 0
            self.precio_acumulado = precio_acumulado
            # Información del proceso de pago
            balance = 0
            self.balance = balance

            # Áreas
            # Área de Productos
            area_compras_xi, area_compras_yi, area_compras_xf, area_compras_yf = self.calcular_area(frame,
                                                                                                    0.0351,
                                                                                                    0.0486,
                                                                                                    0.7539,
                                                                                                    0.9444)
            # Dibujar
            color = (0, 255, 0)
            texto_compras = 'Productos'
            tamano_compras, grosor_compras = 1, 2
            frame = self.dibujar_area(frame, color, area_compras_xi, area_compras_yi, area_compras_xf, area_compras_yf)
            frame = self.dibujar_texto(frame, color, texto_compras, area_compras_xi, area_compras_yi - 5, tamano_compras, grosor_compras)

            # Área de pago
            area_pago_xi, area_pago_yi, area_pago_xf, area_pago_yf = self.calcular_area(frame,
                                                                                        0.7739,
                                                                                        0.0486,
                                                                                        0.9649,
                                                                                        0.6050)
            # Dibujar
            color = (0, 0, 0)
            texto_pago = 'Pago'
            tamano_pago, grosor_pago = 1, 2
            frame = self.dibujar_linea(frame, color, area_pago_xi, area_pago_yi, area_pago_xi,
                                       int((area_pago_yi + area_pago_yf) / 2))
            frame = self.dibujar_linea(frame, color, area_pago_xi, area_pago_yi, int((area_pago_xi + area_pago_xf) / 2),
                                       area_pago_yi)
            frame = self.dibujar_linea(frame, color, area_pago_xf, int((area_pago_yi + area_pago_yf) / 2), area_pago_xf,
                                       area_pago_yf)
            frame = self.dibujar_linea(frame, color, int((area_pago_xi + area_pago_xf) / 2), area_pago_yf, area_pago_xf,
                                       area_pago_yf)
            frame = self.dibujar_texto(frame, color, texto_pago, area_pago_xf - 100, area_compras_yi + 10, tamano_pago,
                                       grosor_pago)

            # Área de la lista
            area_lista_xi, area_lista_yi, area_lista_xf, area_lista_yf = self.calcular_area(frame, 0.7739, 0.6250,
                                                                                            0.9649, 0.9444)
            # Dibujar
            texto_lista = 'Precios'
            tamano_lista, grosor_lista = 0.65, 2
            frame = self.dibujar_linea(frame, color, area_lista_xi, area_lista_yi, area_lista_xi, area_lista_yf)
            frame = self.dibujar_linea(frame, color, area_lista_xi, area_lista_yi, area_lista_xf, area_lista_yi)
            frame = self.dibujar_linea(frame, color, area_lista_xi + 30, area_lista_yi + 30, area_lista_xf - 30,
                                       area_lista_yi + 30)
            frame = self.dibujar_texto(frame, color, texto_lista, area_lista_xi + 55, area_lista_yi + 30, tamano_lista,
                                       grosor_lista)

            # Predicción de Objetos
            frame = self.modelo_prediccion(clean_frame, frame, self.modeloObjetos, clase=0)
            # Predicción de Billetes
            frame = self.modelo_prediccion(clean_frame, frame, self.modeloBilletes, clase=1)

            # Mostrar Precio Acumulado
            texto_precio = f'Total a pagar: {self.precio_acumulado}$'
            frame = self.dibujar_texto(frame, (0, 0, 0), texto_precio, area_lista_xi + 10, area_lista_yf, 0.60, 2,
                                       fondo=False)
            # Mostrar Balance Total
            texto_balance = f'Saldo ingresado: {self.balance_total}$'
            frame = self.dibujar_texto(frame, (0, 0, 0), texto_balance, area_lista_xi + 10, area_lista_yf + 30, 0.60,
                                       2, fondo=False)
            # Pago
            frame = self.dibujar_texto(frame, (0, 0, 0), self.pago, area_lista_xi + - 300, area_lista_yf + 30, 0.60,
                                       2, fondo=False)

            # Mostrar
            cv2.imshow("Tienda IA", frame)

            # Balance
            if t == 83 or t == 115:  # Teclas 'S' o 's'
                self.balance_total += self.balance
                self.balance = 0
            # Pago
            if t == 80 or t == 112:  # Teclas 'P' o 'p'
                self.pago = self.proceso_pago(self.precio_acumulado, self.balance_total)
            # Salir
            if t == 27:  # Tecla 'ESC'
                break

        # Liberar
        self.cap.release()
        cv2.destroyAllWindows()
