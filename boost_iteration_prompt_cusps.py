import os

import numpy as np
import tqdm

from sashimi_c import subhalo_observables, subhalo_properties


def main():
    """Generate the historical prompt-cusp boost lookup tables."""
    f_surv = 1.0
    f_surv_stripped = 1.0
    n_values = [0, 1, 2, 3, 4, 5]

    output_directory = "data/prompt_cusps/boost"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    sh = subhalo_properties()

    list_za = np.arange(0.0, 7.0, 0.1)
    ma_max = sh.Mzi(1.0e16, list_za)
    list_ma = np.logspace(-5, np.log10(ma_max / sh.Msun), 22).T * sh.Msun

    np.savetxt(f"{output_directory}/za.txt", list_za)
    np.savetxt(f"{output_directory}/ma.txt", list_ma)

    list_fsh = np.empty_like(list_ma)
    list_Bsh = np.empty_like(list_ma)
    list_Bcusp_dressed = np.empty_like(list_ma)
    list_Bcusp_naked = np.empty_like(list_ma)
    list_Ncusp_dressed = np.empty_like(list_ma)
    list_Ncusp_naked = np.empty_like(list_ma)

    for n in n_values:
        print(f"Running for n = {n}")
        for i, za in tqdm.tqdm(enumerate(list_za), total=len(list_za)):
            for j, ma in enumerate(list_ma[i]):
                sh = subhalo_observables(
                    ma / sh.Msun,
                    za,
                    M0_at_redshift=True,
                    prompt_cusps=True,
                    logmamin=-9,
                    ct_th=0.0,
                    N_ma=600,
                )
                fsh = sh.mass_fraction()
                (
                    Bsh,
                    Bcusp_dressed,
                    Bcusp_naked,
                    _,
                    Ncusp_dressed,
                    Ncusp_naked,
                ) = sh.annihilation_boost_factor_prompt_cusps(
                    n=n,
                    f_surv=f_surv,
                    f_surv_stripped=f_surv_stripped,
                )

                list_fsh[i, j] = fsh
                list_Bsh[i, j] = Bsh
                list_Bcusp_dressed[i, j] = Bcusp_dressed
                list_Bcusp_naked[i, j] = Bcusp_naked
                list_Ncusp_dressed[i, j] = Ncusp_dressed
                list_Ncusp_naked[i, j] = Ncusp_naked

        if not os.path.exists(f"{output_directory}/fsh.txt"):
            np.savetxt(f"{output_directory}/fsh.txt", list_fsh)

        suffix = f"{n}_{f_surv:.1f}_{f_surv_stripped:.1f}.txt"
        np.savetxt(f"{output_directory}/Bsh_{suffix}", list_Bsh)
        np.savetxt(
            f"{output_directory}/Bcusp_dressed_{suffix}",
            list_Bcusp_dressed,
        )
        np.savetxt(
            f"{output_directory}/Bcusp_naked_{suffix}",
            list_Bcusp_naked,
        )
        np.savetxt(
            f"{output_directory}/Ncusp_dressed_{suffix}",
            list_Ncusp_dressed,
        )
        np.savetxt(
            f"{output_directory}/Ncusp_naked_{suffix}",
            list_Ncusp_naked,
        )


if __name__ == "__main__":
    main()
