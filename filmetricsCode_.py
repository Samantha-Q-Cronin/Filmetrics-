import numpy as np
from scipy.optimize import least_squares
from tmm import coh_tmm

print("PYTHON STARTED")

# Load data
background_array = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\background.txt")
silicon_array = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\scan_1.txt")

intensity_array = silicon_array.copy()
intensity_array[:, 1] = silicon_array[:, 1] - background_array[:, 1]

reflectance_array = np.loadtxt(
    r"C:\Users\AttoPC\Desktop\Filmetrics\sire.txt",
    delimiter="\t"
)


N = min(len(intensity_array), 3085)

interpolatedReflectance_array = np.zeros(N)

for i in range(N):
    data1 = intensity_array[i, 0]

    if reflectance_array[0, 0] < data1 < reflectance_array[-1, 0]:

        for k in range(len(reflectance_array) - 1):

            if reflectance_array[k, 0] <= data1 <= reflectance_array[k + 1, 0]:

                x1 = reflectance_array[k, 0]
                x2 = reflectance_array[k + 1, 0]
                y1 = reflectance_array[k, 1]
                y2 = reflectance_array[k + 1, 1]

                interpolatedReflectance_array[i] = (
                    (y2 - y1) / (x2 - x1)
                ) * (data1 - x1) + y1
                break

reflectance_interp_array = np.zeros((N, 2))

for i in range(N):
    reflectance_interp_array[i, 0] = intensity_array[i, 0]
    reflectance_interp_array[i, 1] = interpolatedReflectance_array[i]

correctionFactor_array = np.zeros(N)

for i in range(N):
    if reflectance_interp_array[i, 1] != 0:
        correctionFactor_array[i] = intensity_array[i, 1] / (reflectance_interp_array[i, 1] + 1e-12)

sampleIntensity_array = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\scan_2.txt")

sampleReflectance_array = np.zeros((N, 2))

for i in range(N):
    sampleReflectance_array[i, 0] = sampleIntensity_array[i, 0]
    sampleReflectance_array[i, 1] = (
        sampleIntensity_array[i, 1] / (correctionFactor_array[i] + 1e-12)
    ) * 100

np.savetxt(
    r"C:\Users\AttoPC\Desktop\Filmetrics\measured reflectance.txt",
    sampleReflectance_array,
    delimiter="\t",
    comments=""
)

print("Saved: measured reflectance.txt")


with open(r"C:\Users\AttoPC\Desktop\Filmetrics\material names.txt", "r") as f:
    material_list = [line.strip() for line in f if line.strip()]
	
def tmm1(n_list, d_list, wl):
    return coh_tmm('s', np.array(n_list), np.array(d_list), 0, wl)

# Load material refractive index database files
data_si3n4 = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Si3N4.txt")
data_sio2 = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\SiO2.txt")
data_si = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Si.txt")
data_tio2 = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\TiO2.txt")
data_al2o3 = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Al2O3.txt")
data_PMMA950k = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\PMMA950K.txt")
data_ta2o5 = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Ta2O5.txt")
data_au = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Au.txt")
data_ag = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Ag.txt")

# wavelength step size (1nm) to capture thin film phases
wavelengths = np.arange(500, 1000, 1)  
num_layers = len(material_list)

#read the index of refraction of each material
n_Si3N4_arr = np.interp(wavelengths, data_si3n4[:, 0], data_si3n4[:, 1])
n_SiO2_arr = np.interp(wavelengths, data_sio2[:, 0], data_sio2[:, 1])
n_Si_arr = np.interp(wavelengths, data_si[:, 0], data_si[:, 1])
n_TiO2_arr = np.interp(wavelengths, data_tio2[:, 0], data_tio2[:, 1])
n_Al2O3_arr = np.interp(wavelengths, data_al2o3[:, 0], data_al2o3[:, 1])
n_PMMA950K_arr = np.interp(wavelengths, data_PMMA950k[:, 0], data_PMMA950k[:, 1])
n_Ta2O5_arr = np.interp(wavelengths, data_ta2o5[:, 0], data_ta2o5[:, 1])
n_Au_arr = np.interp(wavelengths, data_au[:, 0], data_au[:, 1])
n_Ag_arr = np.interp(wavelengths, data_ag[:, 0], data_ag[:, 1])
# material database
material_database = {
    "Si3N4": n_Si3N4_arr,
    "SiO2": n_SiO2_arr,
    "Si": n_Si_arr,
    "TiO2": n_TiO2_arr,
    "Al2O3": n_Al2O3_arr,
	"PMMA950K":n_PMMA950K_arr,
	"Ta2O5":n_Ta2O5_arr,
	"Au":n_Au_arr,
	"Ag":n_Ag_arr
}


# build n_matrix based on user-selected order
n_matrix = []

for mat in material_list:

    if mat not in material_database:
        raise ValueError(f"Material {mat} not found in database")

    n_matrix.append(material_database[mat])


n_matrix = np.array(n_matrix)

# load raw experimental measurement data
measured_data = sampleReflectance_array
raw_wavelengths = measured_data[:, 0]
raw_reflectance = measured_data[:, 1]

# interpolate raw experimental signal onto identical target calculation grid
R_measured = np.interp(wavelengths, raw_wavelengths, raw_reflectance)
if np.max(R_measured) <= 1.0:
    R_measured = R_measured * 100  # Scale up automatically if parsing 0.0-1.0 profile

R_measured_mean = np.mean(R_measured)
denom_variance = np.sum((R_measured - R_measured_mean) ** 2)

#store user's guess
user_guesses = []


# read thickness guesses from file
with open(r"C:\Users\AttoPC\Desktop\Filmetrics\thicknessGuess.txt", "r") as f:
    guess_list = [float(line.strip()) for line in f if line.strip()]

for i, mat in enumerate(material_list):
    guess = guess_list[i]
    user_guesses.append(guess)
	
# read user selected substrate
with open(r"C:\Users\AttoPC\Desktop\Filmetrics\substrate.txt", "r") as f:
     substrate_name = f.read().strip()

if substrate_name == "Silicon":
	raw_n = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Si.txt")
	n_substrate = np.interp(wavelengths, raw_n[:, 0], raw_n[:, 1])

elif substrate_name == "Quartz":
	raw_n = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Quartz.txt")
	n_substrate = np.interp(wavelengths, raw_n[:, 0], raw_n[:, 1])

elif substrate_name == "Sapphire":
	raw_n = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Sapphire.txt")
	n_substrate = np.interp(wavelengths, raw_n[:, 0], raw_n[:, 1])
	
elif substrate_name == "Copper":
	raw_n = np.loadtxt(r"C:\Users\AttoPC\Desktop\Filmetrics\index of refraction data\Cu.txt")
	n_substrate = np.interp(wavelengths, raw_n[:, 0], raw_n[:, 1])
	
print(f"Detected stack configuration: light -> {' -> '.join(material_list)} -> {substrate_name} Substrate")

#vectorized calculation residual vector objective function
def residual_function(params):
    thicknesses = params[:-1]
    scale_factor = params[-1]
    d_list = [np.inf] + list(thicknesses) + [np.inf]
   
    R_spectrum = np.empty(len(wavelengths))
	
    for idx, wl in enumerate(wavelengths):
        # construct the runtime layer index order vector
        middle_layers_n = list(n_matrix[:, idx])
        # get substrate index based on file selection
        n_sub = n_substrate[idx]
        # Air + layers + substrate
        n_list = [1.0] + middle_layers_n + [n_sub]
        R_spectrum[idx] = tmm1(n_list, d_list, wl)['R'] * 100
       
    return R_measured - (R_spectrum * scale_factor)

# bounded multi-start local grid phase search
print("\nScanning around your guess...")
import itertools

search_percent = 0.15

offset_lists = []

for guess in user_guesses:

    search_range = max(20, guess * search_percent)

    offsets = np.linspace(
        guess - search_range,
        guess + search_range,
        5
    )

    offset_lists.append(offsets)

best_gof = -np.inf
best_params = None

lower_limits = [0.1] * num_layers + [0.80]
upper_limits = [10000.0] * num_layers + [1.20]

# multi-start matrix exploration loop
for trial_values in itertools.product(*offset_lists):

    current_seeds = list(trial_values)

    current_seeds.append(1.0)  # intensity scale factor

    seed_state = np.array(current_seeds, dtype=float)

    seed_state = np.clip(
        seed_state,
        lower_limits,
        upper_limits
    )

    res = least_squares(
        residual_function,
        seed_state,
        method='trf',
        bounds=(lower_limits, upper_limits),
        max_nfev=5
    )

    num_res = np.sum(res.fun ** 2)

    current_gof = (
        1.0 - (num_res / denom_variance)
    ) * 100

    if current_gof > best_gof:
        best_gof = current_gof
        best_params = res.x
		
# final convergence polish loop run 
    final_result = least_squares(
    residual_function,
    best_params,
    method='trf',
    bounds=(lower_limits, upper_limits)
)

final_thicknesses = final_result.x[:-1]
final_scale = final_result.x[-1]
final_gof = (1.0 - (np.sum(final_result.fun ** 2) / denom_variance)) * 100

for i, thick in enumerate(final_thicknesses):
    print(f"Layer {i+1} ({material_list[i]}) Calculated Thickness: {thick:.2f} nm")
print(f"Final True Goodness of Fit (GOF): {final_gof:.2f}%\n")

# compute the R_best
R_best = np.empty(len(wavelengths))

for idx, wl in enumerate(wavelengths):

    middle_layers_n = list(n_matrix[:, idx])
    n_list = [1.0] + middle_layers_n + [n_substrate[idx]]

    d_list = [np.inf] + list(final_thicknesses) + [np.inf]

    Rs = coh_tmm('s', n_list, d_list, 0, wl)['R']
    Rp = coh_tmm('p', n_list, d_list, 0, wl)['R']

    R_best[idx] = 0.5 * (Rs + Rp) * 100
	
out = np.column_stack((wavelengths, R_best))

np.savetxt(
    r"C:\Users\AttoPC\Desktop\Filmetrics\calculated reflectance.txt",
    out,
    delimiter="\t",
    comments=""
)
