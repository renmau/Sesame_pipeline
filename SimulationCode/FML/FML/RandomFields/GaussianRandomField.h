#ifndef GAUSSIANRANDOMFIELD_HEADER
#define GAUSSIANRANDOMFIELD_HEADER
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#ifdef USE_GSL
#include <FML/Spline/Spline.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/RandomGenerator/RandomGenerator.h>

namespace FML {
    namespace RANDOM {

        //=================================================================================
        ///
        /// Generate a gaussian random field in real or fourier space from a given
        /// power-spectrum and a source of random numbers. Templated on dimension.
        ///
        /// fix_amplitude means we fix the amplitude of each mode (so not
        /// really a gaussian random field anymore, but very similar in many respects and
        /// it gets around sample variance) so only phases are random. This option guarantees
        /// more or less that the power-spectrum is the same as the input power-spectrum
        ///
        //=================================================================================

        namespace GAUSSIAN {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;

            //=================================================================================
            /// This generates gaussian white noise in real-space. It is normalized such that
            /// the power-spectrum of the field is unity, so the amplitude in real-space is
            /// \f$ {\rm Nmesh}^{N/2} \f$
            ///
            /// @tparam N The dimension we are it
            ///
            /// @param[out] grid The real grid we generate.
            /// @param[in] rng The random generator.
            ///
            //=================================================================================
            template <int N>
            void generate_white_noise_field_real(FFTWGrid<N> & grid, RandomGenerator * rng) {

                auto Nmesh = grid.get_nmesh();
                auto Local_nx = grid.get_local_nx();
                auto Local_x_start = grid.get_local_x_start();

                // The norm is to get unit power-spectrum
                const double norm = std::pow(grid.get_nmesh(), N / 2.0);

                // Communicate the seeds for each slice over tasks to ensure we get the
                // same result no matter how many tasks we use
                std::vector<unsigned int> seedtable(Nmesh, 0);
                if (FML::ThisTask == 0) {
                    for (int i = 0; i < Nmesh; i++) {
                        seedtable[i] = (unsigned int)(INT_MAX * rng->generate_uniform());
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, seedtable.data(), Nmesh, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
#endif
                // Make random generators for each slice to ensure we get the same
                // result no matter how many threads we use
                std::vector<std::shared_ptr<RandomGenerator>> rngs(Local_nx);
                for (int i = 0; i < Local_nx; i++)
                    rngs[i] = rng->clone();

#ifdef USE_OMP
#pragma omp parallel for
#endif
                // Loop over slices
                for (int slice = 0; slice < Local_nx; slice++) {
                    // Seed the rng
                    auto _rng = rngs[slice];
                    _rng->set_seed(seedtable[slice + Local_x_start]);

                    // Loop over cells
                    for (auto & real_index : grid.get_real_range(slice, slice + 1)) {
                        auto value = _rng->generate_normal() * norm;
                        grid.set_real_from_index(real_index, value);
                        if (slice + Local_x_start == 0)
                            std::cout << value << "\n";
                    }
                }
            }

            //=================================================================================
            ///
            /// @tparam N The dimension we are it
            ///
            /// @param[out] grid The real grid we generate.
            /// @param[in] rng The random number generator.
            /// @param[in] Pofk_of_kBox_over_volume This is \f$ P(kB) / V \f$ where \f$ kB \f$ is the dimesnionless
            /// wavenumber where \f$ B \f$ is the boxsize) and \f$ V = B^{\rm N} \f$ is the volume of the box.
            /// @param[in] fix_amplitude If true then we only draw phases and set \f$ |\delta(k)| \f$ directly from the
            /// input power-spectrum.
            ///
            //=================================================================================
            template <int N>
            void generate_gaussian_random_field_real(FFTWGrid<N> & grid,
                                                     RandomGenerator * rng,
                                                     std::function<double(double)> Pofk_of_kBox_over_volume,
                                                     bool fix_amplitude);

            //=================================================================================
            ///
            /// @tparam N The dimension we are it
            ///
            /// @param[out] grid The fourier grid we generate.
            /// @param[in] rng The random number generator.
            /// @param[in] Pofk_of_kBox_over_volume This is \f$ P(kB) / V \f$ where \f$ kB \f$ is the dimesnionless
            /// wavenumber where \f$ B \f$ is the boxsize and \f$ V = B^{\rm N} \f$ is the volume of the box.
            /// @param[in] fix_amplitude If true then we only draw phases and set \f$ |\delta(k)| \f$ directly from the
            /// input power-spectrum.
            ///
            //=================================================================================
            template <int N>
            void generate_gaussian_random_field_fourier(FFTWGrid<N> & grid,
                                                        RandomGenerator * rng,
                                                        std::function<double(double)> Pofk_of_kBox_over_volume,
                                                        bool fix_amplitude);

            //=================================================================================

            template <int N>
            void generate_gaussian_random_field_real(FFTWGrid<N> & grid,
                                                     RandomGenerator * rng,
                                                     std::function<double(double)> Pofk_of_kBox_over_volume,
                                                     bool fix_amplitude) {

                generate_gaussian_random_field_fourier(grid, rng, Pofk_of_kBox_over_volume, fix_amplitude);

                // Fourier transform
                grid.fftw_c2r();
            }

            template <int N>
            void generate_gaussian_random_field_fourier(FFTWGrid<N> & grid,
                                                        RandomGenerator * rng,
                                                        std::function<double(double)> Pofk_of_kBox_over_volume,
                                                        bool fix_amplitude) {

                using IndexIntType = long long int;

                // We require an allocated grid, a random number generator and a power-spectrum to run
                assert_mpi(grid.get_nmesh() > 0,
                           "[generate_gaussian_random_field_fourier] Grid is not allocated. Nmesh is zero\n");
                assert_mpi(Pofk_of_kBox_over_volume.operator bool(),
                           "[generate_gaussian_random_field_fourier] PowerSpectrum function not callable\n");

                // Zero out grid
                grid.fill_fourier_grid(0.0);

                int Nmesh = grid.get_nmesh();
                auto Local_nx = grid.get_local_nx();
                auto Local_x_start = grid.get_local_x_start();
                auto * cdelta = grid.get_fourier_grid();

                const IndexIntType NmeshTotTotal = FML::power(Nmesh, N - 1) * (Nmesh / 2 + 1);
                const IndexIntType factor = NmeshTotTotal / Nmesh;

                // Set up seeds for the random number generator
                // and ensure that all tasks use the same seed
                IndexIntType num_seeds = FML::power(Nmesh, N - 1);
                std::vector<unsigned int> seedtable(num_seeds, 0);
                if (FML::ThisTask == 0) {
                    for (IndexIntType i = 0; i < num_seeds; i++) {
                        seedtable[i] = (unsigned int)(INT_MAX * rng->generate_uniform());
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, seedtable.data(), num_seeds, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
#endif

                // Generate gaussian random field in k-space
                std::vector<int> coord(N, 0), mirrorcoord(N, 0);
                const auto imin_local = Local_x_start;
                const auto imax_local = Local_x_start + Local_nx;
                std::array<double, N> kvec;
                double kmag;

                // We loop over all cells in the global grid. We can cut this down
                // with some work by adjusting only to cells where coord or mirrorcoord
                // is in bounds
                for (IndexIntType ind = 0; ind < NmeshTotTotal; ind++) {
                    coord[0] = ind / factor;
                    mirrorcoord[0] = coord[0] == 0 ? 0 : Nmesh - coord[0];

                    // Only modes that are on current CPU
                    if ((coord[0] >= imin_local and coord[0] < imax_local) or
                        (mirrorcoord[0] >= imin_local and mirrorcoord[0] < imax_local)) {

                        // Compute the rest of the coords
                        coord[N - 1] = ind % (Nmesh / 2 + 1);
                        mirrorcoord[N - 1] = coord[N - 1] == 0 ? 0 : Nmesh - coord[N - 1];
                        IndexIntType n = Nmesh / 2 + 1;
                        for (int idim = N - 2; idim >= 1; idim--, n *= Nmesh) {
                            coord[idim] = (ind / n) % Nmesh;
                            mirrorcoord[idim] = coord[idim] == 0 ? 0 : Nmesh - coord[idim];
                        }

                        // Set seed for every new slice in the last direction
                        if (coord[N - 1] == 0)
                            rng->set_seed(seedtable[ind / (Nmesh / 2 + 1)]);

                        // Gaussian random number
                        double phase = rng->generate_uniform() * 2 * M_PI;
                        double norm = 1.0;
                        if (not fix_amplitude) {
                            norm = rng->generate_uniform();
                            norm = norm > 0.0 ? -std::log(norm) : 1.0;
                        }

                        // Skip modes that are zero or otherwise fixed by symmetry
                        // 1) When all coord are 0 (the DC mode)
                        // 2) When the mode and mirror mode is the same
                        // 3) Skip the mirror modes so we don't assign them twice
                        if (ind == 0)
                            continue;

                        for (int idim = 0; idim < N; idim++) {
                            if (coord[idim] == Nmesh / 2)
                                goto endloop;
                        }

                        if (coord[0] == 0 and coord[N - 1] == 0) {
                            for (int idim = 1; idim < N - 1; idim++) {
                                if (coord[idim] >= Nmesh / 2)
                                    goto endloop;
                            }
                        }

                        // Compute local index of mode
                        IndexIntType index = coord[0] - imin_local;
                        for (int idim = 1; idim < N - 1; idim++) {
                            index = index * Nmesh + coord[idim];
                        }
                        index = index * (Nmesh / 2 + 1) + coord[N - 1];

                        // The wave-vector and norm of current mode (norm in units of 1/Box)
                        grid.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);

                        // Assign the field. Note kmag is dimensionless here. Units taken care of in Powerspectrum
                        double delta_norm = std::sqrt(norm * Pofk_of_kBox_over_volume(kmag));
                        std::complex<double> delta = delta_norm * std::exp(std::complex<double>(0, 1) * phase);
                        std::complex<double> delta_conj = std::conj(delta);

                        // The case [0 < k < Nmesh/2] for all the local model
                        if (coord[N - 1] > 0 and imin_local <= coord[0] and coord[0] < imax_local) {
                            cdelta[index] = delta;
                            continue;
                        } else if (coord[N - 1] > 0)
                            continue;

                        // Compute local mirror index
                        IndexIntType mirrorindex = mirrorcoord[0] - imin_local;
                        for (int idim = 1; idim < N - 1; idim++) {
                            mirrorindex = mirrorindex * Nmesh + mirrorcoord[idim];
                        }
                        mirrorindex = mirrorindex * (Nmesh / 2 + 1) + mirrorcoord[N - 1];

                        // The case [iglobal = 0] and [k = 0]
                        if (coord[0] == 0) {
                            if (imin_local == 0) {
                                cdelta[index] = delta;
                                cdelta[mirrorindex] = delta_conj;
                            }
                            continue;
                        }

                        // The case [0 < i < Nmesh/2], [0 < j < Nmesh] and [k = 0]
                        if (imin_local <= coord[0] and coord[0] < imax_local) {
                            cdelta[index] = delta;
                        }

                        // The mirror mode
                        if (imin_local <= mirrorcoord[0] and mirrorcoord[0] < imax_local) {
                            cdelta[mirrorindex] = delta_conj;
                        }
                    }
                endloop:;
                }
            }

            // Specialization for N=3 to have the random seeds agree with what is done in N-GenIC / PICOLA / MPICOLA
            // allowing to create the same IC as them given the same seed
            template <>
            void generate_gaussian_random_field_fourier(FFTWGrid<3> & grid,
                                                        RandomGenerator * rng,
                                                        std::function<double(double)> Pofk_of_kBox_over_volume,
                                                        bool fix_amplitude) {

                // We require an allocated grid, a random number generator and a power-spectrum to run
                assert_mpi(grid.get_nmesh() > 0,
                           "[generate_gaussian_random_field_fourier<3>] Grid is not allocated. Nmesh is zero\n");
                assert_mpi(Pofk_of_kBox_over_volume.operator bool(),
                           "[generate_gaussian_random_field_fourier<3>] PowerSpectrum function not callable\n");

                // Zero out grid
                grid.fill_fourier_grid(0.0);

                int Nmesh = grid.get_nmesh();
                auto Local_nx = grid.get_local_nx();
                auto Local_x_start = grid.get_local_x_start();
                auto * cdelta = grid.get_fourier_grid();

                const auto imin_local = Local_x_start;
                const auto imax_local = Local_x_start + Local_nx;

                // Set up seeds for the random number generator
                // The strange way it's done here (instead of just filling the N^2 table directly)
                // is just to keep compatibility with PICOLA which allows for creating exactly the same IC when rng is
                // GSL and we use the same seed
                std::vector<unsigned int> seedtable(Nmesh * Nmesh, 0);
                if (FML::ThisTask == 0)
                    for (int i = 0; i < Nmesh / 2; i++) {
                        for (int j = 0; j < i; j++)
                            seedtable[i * Nmesh + j] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i + 1; j++)
                            seedtable[j * Nmesh + i] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i; j++)
                            seedtable[(Nmesh - 1 - i) * Nmesh + j] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i + 1; j++)
                            seedtable[(Nmesh - 1 - j) * Nmesh + i] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i; j++)
                            seedtable[i * Nmesh + (Nmesh - 1 - j)] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i + 1; j++)
                            seedtable[j * Nmesh + (Nmesh - 1 - i)] = (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i; j++)
                            seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] =
                                (unsigned int)(INT_MAX * rng->generate_uniform());
                        for (int j = 0; j < i + 1; j++)
                            seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] =
                                (unsigned int)(INT_MAX * rng->generate_uniform());
                    }

#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, seedtable.data(), Nmesh * Nmesh, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
#endif
                
                // If we use GSL make a spline (std::function can be slow) otherwise this is just a copy
                // of the function itself
                auto Pofk_of_kBox_over_volume_spline = grid.make_fourier_spline(Pofk_of_kBox_over_volume, "P(k)/V");

                // Generate gaussian random field in k-space
                std::array<double, 3> kvec;
                for (int i = 0; i < Nmesh; i++) {
                    int ii = i == 0 ? 0 : Nmesh - i;

                    // Only create modes that belong to current task
                    if ((i >= imin_local and i < imax_local) or (ii >= imin_local and ii < imax_local)) {

                        for (int j = 0; j < Nmesh; j++) {
                            int jj = j == 0 ? 0 : Nmesh - j;
                            rng->set_seed(seedtable[i * Nmesh + j]);

                            for (int k = 0; k < Nmesh / 2 + 1; k++) {
                                size_t coord;

                                // Gaussian random number
                                double phase = rng->generate_uniform() * 2 * M_PI;
                                double norm = 1.0;
                                if (not fix_amplitude) {
                                    norm = rng->generate_uniform();
                                    norm = norm > 0.0 ? -std::log(norm) : 1.0;
                                }

                                // Skip modes that are zero or otherwise fixed by symmetry
                                if (i == Nmesh / 2 or j == Nmesh / 2 or k == Nmesh / 2)
                                    continue;
                                if (i == 0 and j >= Nmesh / 2 and k == 0)
                                    continue;
                                if (i == 0 and j == 0 and k == 0)
                                    continue;
                                if (i >= Nmesh / 2 and k == 0)
                                    continue;

                                // The wave-vector and norm of current mode (norm in units of 1/Box)
                                kvec[0] = i <= Nmesh / 2 ? i : i - Nmesh;
                                kvec[1] = j <= Nmesh / 2 ? j : j - Nmesh;
                                kvec[2] = k <= Nmesh / 2 ? k : k - Nmesh;
                                double kmag =
                                    std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]) * 2.0 * M_PI;

                                // Assign the field. Note kmag is dimensionless here. Units taken care of in
                                // Powerspectrum
                                double delta_norm = std::sqrt(norm * Pofk_of_kBox_over_volume_spline(kmag));

                                std::complex<double> delta = delta_norm * std::exp(std::complex<double>(0, 1) * phase);
                                std::complex<double> delta_conj = std::conj(delta);

                                // The case [0 < k < Nmesh/2] for all the local model
                                if (k > 0 and imin_local <= i and i < imax_local) {
                                    coord = ((i - imin_local) * Nmesh + j) * (Nmesh / 2 + 1) + k;
                                    cdelta[coord] = delta;
                                    continue;
                                } else if (k > 0)
                                    continue;

                                // The case [iglobal = 0], [0 < j < Nmesh/2] and [k = 0]
                                if (i == 0) {
                                    if (imin_local == 0) {
                                        coord = ((i - imin_local) * Nmesh + j) * (Nmesh / 2 + 1) + k;
                                        cdelta[coord] = delta;

                                        // The mirror mode
                                        coord = ((i - imin_local) * Nmesh + jj) * (Nmesh / 2 + 1) + k;
                                        cdelta[coord] = delta_conj;
                                    }
                                    continue;
                                }

                                // The case [0 < i < Nmesh/2], [0 < j < Nmesh] and [k = 0]
                                if (imin_local <= i and i < imax_local) {
                                    coord = ((i - imin_local) * Nmesh + j) * (Nmesh / 2 + 1) + k;
                                    cdelta[coord] = delta;
                                }

                                // The mirror mode
                                if (imin_local <= ii and ii < imax_local) {
                                    coord = ((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k;
                                    cdelta[coord] = delta_conj;
                                }
                            }
                        }
                    }
                }
            }

        } // namespace GAUSSIAN
    }     // namespace RANDOM
} // namespace FML
#endif
