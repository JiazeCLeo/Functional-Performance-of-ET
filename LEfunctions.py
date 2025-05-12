import pandas as pd

def fun_LEPT(df, Qn_col, T_col, P_col, GH_col, e_vsat_col):
    # Constants
    Cp = 1005  # J/(kg K), Specific heat capacity at constant pressure
    la = 2260000  # J/kg, Latent heat of vaporization
    e = 0.622  # Water vapor mixing ratio
    alpha = 1  # Alpha coefficient
    
    # Extract columns from the DataFrame
    Qn_25 = df[Qn_col]
    T_25 = df[T_col]
    GH = df[GH_col]
    P = df[P_col]
    e_vsat = df[e_vsat_col]

    # Calculate psychrometric constant (psyC)
    psyC = Cp * P / e / la

    # Calculate delta (slope of the saturation vapor pressure curve)
    delta = 4098 * e_vsat / ((T_25 + 237.3) ** 2)

    # Calculate simulated latent heat flux (LEsim)
    LEPT = alpha * delta * (Qn_25 - GH) / (delta + psyC)
    
    return LEPT

def fun_LESFE(df, Qn_col, RH_col, T_col, e_vsat_col, Airpressure_col):

    # Constants
    Cp = 1005  # J/(kg K), Specific heat capacity at constant pressure
    Rv = 500  # Specific gas constant for water vapor (J/(kg K))
    la = 2 * 10**6  # J/kg, Latent heat of vaporization
    
    # Extract columns from the DataFrame
    Qn_25 = df[Qn_col]
    T_25 = df[T_col]
    RH_25 = df[RH_col]
    e_vsat = df[e_vsat_col]
    Airpressures = df[Airpressure_col]

    # Calculate mixing ratio (mvsat)
    mvsat = 0.622 * e_vsat / (Airpressures - e_vsat)

    # Calculate specific humidity (q_25)
    q_25 = RH_25 * mvsat / 100

    # Calculate Bowen ratio for SFE
    Bowen_SFE_25 = (Rv * Cp * (T_25 + 273)**2) / (la**2 * q_25)

    # Calculate LESFE
    LESFE = Qn_25 / (Bowen_SFE_25 + 1)
    
    return LESFE