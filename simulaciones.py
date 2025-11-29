# archivo: sim_fuel_systems_np_optimized_progress.py
# Requisitos: numpy, pandas, matplotlib, tqdm
# pip install numpy pandas matplotlib tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import math
import unicodedata

# ---------------- Parámetros generales ----------------
SEED = 42
np.random.seed(SEED)

# Motor ajustado para moto 125cc
desplazamiento_l = 0.125      # litros (125cc)
densidad_aire = 1.2           # kg/m3 aprox
VE = 0.9                      # volumetric efficiency (0-1)
caudal_inyector_cc_min = 300.0  # cc/min por inyector (ejemplo)
densidad_combustible_kg_l = 0.75  # kg/L (gasolina)
cantidad_inyectores = 1       # inyectores por motor
capacidad_tanque_L = 11.0     # litros

# Mapeo base (RPM x TPS)
bins_rpm = np.linspace(1000, 12000, 25)    # 25 bins -> 24 celdas
bins_tps = np.linspace(0.0, 100.0, 25)     # 25 bins -> 24 celdas

# Rangos por defecto para pisos térmicos (sin tierra helada)
pisos_termicos = {
    1: ("Tierra caliente", (24, 38), (95, 101)),
    2: ("Tierra templada", (17, 24), (90, 101)),
    3: ("Tierra fría", (12, 17), (85, 101)),
    4: ("Páramo", (6, 12), (70, 95)),
}

# Conversión caudal inyector cc/min -> kg/s
def inyector_ccmin_a_kg_s(cc_min, densidad=densidad_combustible_kg_l):
    L_min = cc_min / 1000.0
    kg_min = L_min * densidad
    return kg_min / 60.0

kg_s_inyector = inyector_ccmin_a_kg_s(caudal_inyector_cc_min) * cantidad_inyectores

AFR_objetivo = 14.7  # relación aire/combustible objetivo

# ---------------- Helpers ----------------
def safe_name(s):
    """Convierte nombre a minúsculas, sin tildes ni espacios (para archivos)."""
    s_nf = unicodedata.normalize('NFKD', s)
    s_ascii = ''.join(c for c in s_nf if not unicodedata.combining(c))
    return s_ascii.replace(' ', '_').lower()

def pedir_rangos_por_defecto(nombre_piso, iat_rango, presion_rango):
    print(f"\nHas seleccionado '{nombre_piso}'.")
    print("Los rangos recomendados para este piso térmico son:")
    print(f"  Temperatura de admisión (IAT): {iat_rango[0]}°C a {iat_rango[1]}°C")
    print(f"  Presión ambiente: {presion_rango[0]} kPa a {presion_rango[1]} kPa")
    input("Presiona Enter para aceptar estos rangos y continuar...")
    return iat_rango, presion_rango

# Entrada obligatoria (sin default)
def pedir_entero_obligatorio(mensaje):
    while True:
        entrada = input(mensaje + ": ").strip()
        if entrada == "":
            print("Debes ingresar un número entero positivo (no se aceptan valores por defecto).")
            continue
        try:
            val = int(entrada)
            if val > 0:
                return val
            else:
                print("Por favor ingresa un número entero positivo.")
        except:
            print("Entrada inválida. Intenta de nuevo.")

# ---------------- Conversiones ----------------
def masa_aire_a_combustible(masa_aire_kg_s, afr=AFR_objetivo):
    return masa_aire_kg_s / afr

def masa_combustible_a_pulse_width_ms(masa_comb_kg_s, kg_s_inyector_local=kg_s_inyector):
    if np.isscalar(masa_comb_kg_s):
        if kg_s_inyector_local <= 0:
            return 0.0
    else:
        if kg_s_inyector_local <= 0:
            return np.zeros_like(masa_comb_kg_s)
    pw_s = masa_comb_kg_s / kg_s_inyector_local
    return pw_s * 1000.0

def pulse_ms_a_Lh(pulse_ms, kg_s_inyector_local=kg_s_inyector, densidad=densidad_combustible_kg_l):
    duty = pulse_ms / 1000.0  # seg
    flujo_kg_s = kg_s_inyector_local * duty
    flujo_L_s = np.divide(flujo_kg_s, densidad, out=np.zeros_like(flujo_kg_s), where=(densidad!=0))
    return flujo_L_s * 3600.0  # L/h

# ---------------- Simulaciones vectorizadas (NP-like, mapa 2D) ----------------
def simular_inyeccion_vectorizada(num_muestras, iat_rango, presion_rango, chunk_size=200000):
    """
    Simulación vectorizada con barra de progreso por chunks.
    """
    shape = (len(bins_rpm)-1, len(bins_tps)-1)
    suma_map = np.zeros(shape, dtype=float)
    cuenta_map = np.zeros(shape, dtype=int)
    muestras_list = []

    total_chunks = math.ceil(num_muestras / chunk_size)
    processed = 0

    for _ in tqdm(range(total_chunks), desc="Simulación INYECCIÓN", unit="chunk"):
        n = min(chunk_size, num_muestras - processed)
        if n <= 0:
            break

        rpms = np.random.uniform(1000.0, 12000.0, size=n)
        tps = np.random.uniform(0.0, 100.0, size=n)
        iat = np.random.uniform(iat_rango[0], iat_rango[1], size=n)
        presion = np.random.uniform(presion_rango[0], presion_rango[1], size=n)

        ciclos_por_seg = rpms / 2.0 / 60.0
        vol_por_ciclo_m3 = (desplazamiento_l / 1000.0) * VE
        flujo_vol_m3_s = vol_por_ciclo_m3 * ciclos_por_seg * (tps / 100.0)
        masa_flujo_kg_s = flujo_vol_m3_s * densidad_aire

        corr_temp = 1.0 - 0.01 * (iat - 20.0)
        corr_presion = presion / 101.0
        masa_aire_corr = masa_flujo_kg_s * corr_temp * corr_presion

        masa_comb_req = masa_aire_corr / AFR_objetivo  # kg/s

        with np.errstate(divide='ignore', invalid='ignore'):
            pw_s = masa_comb_req / kg_s_inyector
        pulse_ms = np.where(kg_s_inyector > 0, pw_s * 1000.0, 0.0)

        idx_rpm = np.digitize(rpms, bins_rpm) - 1
        idx_tps = np.digitize(tps, bins_tps) - 1
        idx_rpm = np.clip(idx_rpm, 0, shape[0]-1)
        idx_tps = np.clip(idx_tps, 0, shape[1]-1)

        pos_plana = idx_rpm * shape[1] + idx_tps
        np.add.at(suma_map.ravel(), pos_plana, pulse_ms)
        np.add.at(cuenta_map.ravel(), pos_plana, 1)

        muestras_chunk = np.column_stack((rpms, tps, iat, presion, pulse_ms))
        muestras_list.append(muestras_chunk)

        processed += n

    with np.errstate(divide='ignore', invalid='ignore'):
        mapa_promedio = np.divide(suma_map, np.where(cuenta_map>0, cuenta_map, 1))
    mapa_promedio[cuenta_map==0] = np.nan
    muestras_detalle = np.vstack(muestras_list) if len(muestras_list) > 0 else np.empty((0,5))
    return mapa_promedio, cuenta_map, muestras_detalle

def simular_carburador_vectorizada(num_muestras, iat_rango, presion_rango, choke_temp_thresh=5.0, choke_rich_factor=1.5, chunk_size=200000):
    shape = (len(bins_rpm)-1, len(bins_tps)-1)
    suma_map = np.zeros(shape, dtype=float)
    cuenta_map = np.zeros(shape, dtype=int)
    muestras_list = []

    total_chunks = math.ceil(num_muestras / chunk_size)
    processed = 0

    for _ in tqdm(range(total_chunks), desc="Simulación CARBURADOR", unit="chunk"):
        n = min(chunk_size, num_muestras - processed)
        if n <= 0:
            break

        rpms = np.random.uniform(1000.0, 8000.0, size=n)
        tps = np.random.uniform(0.0, 100.0, size=n)
        iat = np.random.uniform(iat_rango[0], iat_rango[1], size=n)
        presion = np.full(n, 101.0)  # presión fija para carburador (simplificado)

        ciclos_por_seg = rpms / 2.0 / 60.0
        vol_por_ciclo_m3 = (desplazamiento_l / 1000.0) * VE
        flujo_vol_m3_s = vol_por_ciclo_m3 * ciclos_por_seg * (tps / 100.0)
        masa_flujo_kg_s = flujo_vol_m3_s * densidad_aire

        jet_base_flow = 0.5
        jet_effect = jet_base_flow * (1.0 + 0.01 * tps) * (1.0 + 0.0001 * (rpms - 2000.0))
        masa_comb_req = masa_flujo_kg_s / AFR_objetivo * jet_effect

        choke_mask = (iat <= choke_temp_thresh)
        if choke_rich_factor != 1.0:
            masa_comb_req = masa_comb_req * np.where(choke_mask, choke_rich_factor, 1.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            pw_s = masa_comb_req / kg_s_inyector
        pulse_ms = np.where(kg_s_inyector > 0, pw_s * 1000.0, 0.0)

        idx_rpm = np.digitize(rpms, bins_rpm) - 1
        idx_tps = np.digitize(tps, bins_tps) - 1
        idx_rpm = np.clip(idx_rpm, 0, shape[0]-1)
        idx_tps = np.clip(idx_tps, 0, shape[1]-1)

        pos_plana = idx_rpm * shape[1] + idx_tps
        np.add.at(suma_map.ravel(), pos_plana, pulse_ms)
        np.add.at(cuenta_map.ravel(), pos_plana, 1)

        muestras_chunk = np.column_stack((rpms, tps, iat, presion, pulse_ms))
        muestras_list.append(muestras_chunk)

        processed += n

    with np.errstate(divide='ignore', invalid='ignore'):
        mapa_promedio = np.divide(suma_map, np.where(cuenta_map>0, cuenta_map, 1))
    mapa_promedio[cuenta_map==0] = np.nan
    muestras_detalle = np.vstack(muestras_list) if len(muestras_list) > 0 else np.empty((0,5))
    return mapa_promedio, cuenta_map, muestras_detalle

# ---------------- Consumo y kilometraje ----------------
def calcular_consumo_tanque(muestras_detalle, duracion_seg=3600):
    if muestras_detalle is None or len(muestras_detalle) == 0:
        return 0.0, 0.0
    pulse_ms = muestras_detalle[:, 4]  # ms por muestra
    Lh_por_muestra = pulse_ms_a_Lh(pulse_ms, kg_s_inyector)
    consumo_medio_Lh = np.nanmean(Lh_por_muestra)
    if np.isnan(consumo_medio_Lh) or consumo_medio_Lh <= 0:
        return 0.0, 0.0
    horas_por_tanque = capacidad_tanque_L / consumo_medio_Lh
    km_por_tanque = horas_por_tanque * 40.0  # velocidad promedio estimada 40 km/h
    return consumo_medio_Lh, km_por_tanque

def estimar_km_por_tanque(mapa_Lh, velocidad_promedio_kmh=40):
    if mapa_Lh is None or np.isnan(mapa_Lh).all():
        return 0.0
    consumo_medio_Lh = np.nanmean(mapa_Lh)
    if consumo_medio_Lh <= 0 or np.isnan(consumo_medio_Lh):
        return 0.0
    horas_por_tanque = capacidad_tanque_L / consumo_medio_Lh
    return horas_por_tanque * velocidad_promedio_kmh

# ---------------- NP-like: demostrar explosión combinatoria ----------------
def demostrar_np_like():
    print("\n--- Demostración de crecimiento NP-like ---")
    print("Al aumentar la resolución del mapa o el número de sensores,")
    print("el número de combinaciones posibles crece exponencialmente.\n")
    print("1) Número de celdas en el mapa (RPM × TPS):")
    for bins in [10, 20, 25, 30, 40]:
        celdas = (bins - 1) * (bins - 1)
        print(f"  {bins} bins -> {celdas} celdas (aprox.)")
    print("\n2) Combinaciones con S sensores y L niveles:")
    sensores = [2, 3, 4, 5]
    niveles = [10, 20, 50, 100]
    for s in sensores:
        print(f"  Con {s} sensores:")
        for l in niveles:
            combinaciones = l ** s
            print(f"    {l}^{s} = {combinaciones:.2e} combinaciones")
    print("\n3) Tiempo de muestreo ilustrativo:")
    print("  Si cada muestra toma 1 ms, y queremos cubrir 1e6 combinaciones:")
    print("  Tiempo = 1e6 ms = 1000 s = ~16.7 minutos")
    print("  Para 1e9 combinaciones = ~11.5 días")

# ---------------- Guardado y gráficos ----------------
def guardar_y_graficar(nombre_piso, inj_map_ms, carb_map_ms, inj_count, carb_count, muestras_inj, muestras_carb, outdir="resultados"):
    nombre_safe = safe_name(nombre_piso)
    os.makedirs(outdir, exist_ok=True)

    inj_Lh = pulse_ms_a_Lh(inj_map_ms, kg_s_inyector)
    carb_Lh = pulse_ms_a_Lh(carb_map_ms, kg_s_inyector)

    diff_Lh = inj_Lh - carb_Lh
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = 100.0 * diff_Lh / np.where(np.abs(carb_Lh) > 1e-12, carb_Lh, np.nan)

    # Guardar mapas
    pd.DataFrame(inj_map_ms).to_csv(os.path.join(outdir, f"injection_map_ms_{nombre_safe}.csv"), index=False, header=False)
    pd.DataFrame(carb_map_ms).to_csv(os.path.join(outdir, f"carb_map_ms_{nombre_safe}.csv"), index=False, header=False)
    pd.DataFrame(inj_Lh).to_csv(os.path.join(outdir, f"injection_map_Lh_{nombre_safe}.csv"), index=False, header=False)
    pd.DataFrame(carb_Lh).to_csv(os.path.join(outdir, f"carb_map_Lh_{nombre_safe}.csv"), index=False, header=False)
    pd.DataFrame(diff_Lh).to_csv(os.path.join(outdir, f"difference_map_Lh_{nombre_safe}.csv"), index=False, header=False)
    pd.DataFrame(pct_diff).to_csv(os.path.join(outdir, f"pct_difference_map_{nombre_safe}.csv"), index=False, header=False)

    # Resumen CSV por piso (una sola fila)
    consumo_inj_Lh, km_inj = calcular_consumo_tanque(muestras_inj)
    consumo_carb_Lh, km_carb = calcular_consumo_tanque(muestras_carb)
    celdas_inj = int(np.nansum(~np.isnan(inj_map_ms)))
    celdas_carb = int(np.nansum(~np.isnan(carb_map_ms)))

    resumen = {
        "nombre": [nombre_safe],
        "consumo_iny_Lh": [consumo_inj_Lh],
        "km_iny": [km_inj],
        "consumo_carb_Lh": [consumo_carb_Lh],
        "km_carb": [km_carb],
        "celdas_iny": [celdas_inj],
        "celdas_carb": [celdas_carb]
    }
    resumen_df = pd.DataFrame(resumen)
    resumen_path = os.path.join(outdir, f"resumen_{nombre_safe}.csv")
    resumen_df.to_csv(resumen_path, index=False)

    print(f"\nGuardados resultados en '{outdir}' para {nombre_piso}")
    print(f"  Consumo inyección (estimado): {consumo_inj_Lh:.4f} L/h ({km_inj:.2f} km/tanque)")
    print(f"  Consumo carburador (estimado): {consumo_carb_Lh:.4f} L/h ({km_carb:.2f} km/tanque)")

    # Graficar mapas (L/h)
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.title(f"Mapa Inyección (ms) - {nombre_piso}")
    plt.imshow(inj_map_ms, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.title(f"Mapa Carburador (ms) - {nombre_piso}")
    plt.imshow(carb_map_ms, origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.title(f"Mapa Inyección (L/h) - {nombre_piso}")
    plt.imshow(inj_Lh, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.title(f"Mapa Carburador (L/h) - {nombre_piso}")
    plt.imshow(carb_Lh, origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.title(f"Diferencia absoluta (L/h) - {nombre_piso}")
    plt.imshow(diff_Lh, origin='lower', aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.title(f"Diferencia relativa (%) - {nombre_piso}")
    plt.imshow(pct_diff, origin='lower', aspect='auto', cmap='bwr')
    plt.colorbar()

    plt.tight_layout()
    fig_path = os.path.join(outdir, f"maps_comparacion_{nombre_safe}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"  Imagen generada: {fig_path}")
    print(f"  Resumen guardado: {resumen_path}")

# ---------------- Menú e inicio ----------------
def menu_principal():
    print("Simulación de sistemas de combustible para moto 125cc (NP-like, vectorizado)")
    print("Seleccione un piso térmico:")
    for k in sorted(pisos_termicos.keys()):
        print(f"  {k}) {pisos_termicos[k][0]}")
    print("  5) Todas")
    print("  6) Demostrar NP-like")
    print("  0) Salir")

    opcion = input("Ingrese opción: ").strip()
    if opcion == "0":
        print("Saliendo...")
        exit(0)
    elif opcion == "5":
        return "todas", None, None
    elif opcion == "6":
        return "np", None, None
    else:
        try:
            op_int = int(opcion)
            if op_int in pisos_termicos:
                return "uno", op_int, None
            else:
                print("Opción inválida.")
                return menu_principal()
        except:
            print("Opción inválida.")
            return menu_principal()

def main():
    opcion, piso_seleccionado, _ = menu_principal()

    if opcion == "np":
        demostrar_np_like()
        return

    if opcion == "todas":
        num_inyeccion = pedir_entero_obligatorio("Número de simulaciones para INYECCIÓN (entero positivo)")
        num_carburador = pedir_entero_obligatorio("Número de simulaciones para CARBURADOR (entero positivo)")

        for k in pisos_termicos:
            nombre_piso, iat_rango, presion_rango = pisos_termicos[k]
            print(f"\nEjecutando simulación para piso térmico: {nombre_piso}")

            inj_map_ms, inj_count, muestras_inj = simular_inyeccion_vectorizada(num_inyeccion, iat_rango, presion_rango)
            carb_map_ms, carb_count, muestras_carb = simular_carburador_vectorizada(num_carburador, iat_rango, presion_rango)

            guardar_y_graficar(nombre_piso, inj_map_ms, carb_map_ms, inj_count, carb_count, muestras_inj, muestras_carb)

    elif opcion == "uno":
        nombre_piso, iat_rango, presion_rango = pisos_termicos[piso_seleccionado]
        iat_rango, presion_rango = pedir_rangos_por_defecto(nombre_piso, iat_rango, presion_rango)

        num_inyeccion = pedir_entero_obligatorio("Número de simulaciones para INYECCIÓN (entero positivo)")
        num_carburador = pedir_entero_obligatorio("Número de simulaciones para CARBURADOR (entero positivo)")

        print("\nIniciando simulación de INYECCIÓN (vectorizada)...")
        inj_map_ms, inj_count, muestras_inj = simular_inyeccion_vectorizada(num_inyeccion, iat_rango, presion_rango)
        print("Inyección completada.")

        print("\nIniciando simulación de CARBURADOR (vectorizada)...")
        carb_map_ms, carb_count, muestras_carb = simular_carburador_vectorizada(num_carburador, iat_rango, presion_rango)
        print("Carburador completado.")

        guardar_y_graficar(nombre_piso, inj_map_ms, carb_map_ms, inj_count, carb_count, muestras_inj, muestras_carb)

if __name__ == "__main__":
    main()
