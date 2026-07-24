import os

import numpy as np
import tqdm

from sashimi_c import subhalo_observables, subhalo_properties


def main():
    """Generate the historical annihilation-boost lookup tables."""
    n_values = [0, 1, 2, 3]

    if not os.path.exists("data/boost"):
        os.makedirs("data/boost", exist_ok=True)

    sh = subhalo_properties()

    list_za = np.arange(0.0, 7.0, 0.1)
    ma_max = sh.Mzi(1.0e16, list_za)
    list_ma = np.logspace(-5, np.log10(ma_max / sh.Msun), 22).T * sh.Msun

    np.savetxt("data/boost/za.txt", list_za)
    np.savetxt("data/boost/ma.txt", list_ma)

    list_fsh = np.empty_like(list_ma)
    list_Bsh = np.empty_like(list_ma)

    for n in n_values:
        print(f"Running for n = {n}")
        for i, za in tqdm.tqdm(enumerate(list_za), total=len(list_za)):
            for j, ma in enumerate(list_ma[i]):
                sh = subhalo_observables(
                    ma / sh.Msun,
                    za,
                    M0_at_redshift=True,
                )
                fsh = sh.mass_fraction()
                Bsh, _ = sh.annihilation_boost_factor(n=n)

                list_fsh[i, j] = fsh
                list_Bsh[i, j] = Bsh

        if not os.path.exists("data/boost/fsh.txt"):
            np.savetxt("data/boost/fsh.txt", list_fsh)

        np.savetxt(f"data/boost/Bsh_{n}.txt", list_Bsh)


if __name__ == "__main__":
    main()
